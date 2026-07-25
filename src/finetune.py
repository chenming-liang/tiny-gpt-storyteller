"""
Instruction finetuning script for GPT on Alpaca dataset.
Usage: python src/finetune.py
"""
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup
import wandb

from config import GPTConfig, FinetuneConfig
from model import GPT, count_parameters


def _build_collate_fn(pad_token_id):
    """Dynamic padding within batch for Alpaca finetuning."""
    def collate_fn(batch):
        x_batch, y_batch = zip(*batch)
        x_padded = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=pad_token_id)
        y_padded = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=-1)
        return x_padded, y_padded
    return collate_fn


# ─────────────────────────── Data ───────────────────────────

def format_prompt(example):
    """Format Alpaca example into full text (instruction + response)."""
    if example["input"]:
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )


def build_prompt(instruction, input_text=""):
    """只构造 prompt 部分（不含 output），供生成时使用。"""
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )


class AlpacaDataset(Dataset):
    """Alpaca instruction-following dataset."""

    def __init__(self, tokenizer, max_seq_len: int, mask_instruction: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_instruction = mask_instruction
        self.data = load_dataset("yahma/alpaca-cleaned", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        full_text = format_prompt(example)

        tokens = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_len
        )["input_ids"]

        x = tokens[:-1]
        y = tokens[1:]

        if self.mask_instruction:
            prompt_text = build_prompt(example["instruction"], example["input"])
            prompt_len = len(self.tokenizer(prompt_text, truncation=False)["input_ids"])
            for i in range(min(prompt_len - 1, len(y))):
                y[i] = -1

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─────────────────────────── Finetuning ───────────────────────────

class Finetuner:
    """Handles finetuning loop, loading pretrained weights, and dialog."""

    def __init__(self, model, model_cfg, train_cfg, device, model_name="gpt"):
        self.model = model
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.model_name = model_name

        n_params = count_parameters(self.model)
        print(f"Model parameters: {n_params:.2f}M")

        # optimizer with weight decay groups
        self.optimizer = self._configure_optimizers()

        # cosine scheduler with warmup
        total_steps = train_cfg.max_epochs * train_cfg.max_steps_per_epoch
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),  # 10% warmup
            num_training_steps=total_steps,
        )

        self.scaler = GradScaler(enabled=train_cfg.use_amp)

    def _configure_optimizers(self):
        """Separate params into weight-decay and no-decay groups."""
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if fpn.endswith('bias'):
                    no_decay.add(fpn)
                elif fpn.endswith('weight') and isinstance(m, whitelist):
                    decay.add(fpn)
                elif fpn.endswith('weight') and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter = decay & no_decay
        assert not inter, f"params in both sets: {inter}"
        missing = param_dict.keys() - (decay | no_decay)
        assert not missing, f"params not assigned: {missing}"

        return AdamW(
            [
                {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": self.train_cfg.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            ],
            lr=self.train_cfg.learning_rate,
            betas=(0.9, 0.95),
        )

    def _load_pretrained(self):
        """Load pretrained weights."""
        path = os.path.join('outputs', self.model_name, 'final.pt')
        if not os.path.exists(path):
            print(f"Warning: no pretrained weights found at {path}, starting from scratch")
            return
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state)

    def train_epoch(self, dataloader, epoch: int):
        """Run one epoch of finetuning."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for step, (x, y) in enumerate(dataloader):
            if (
                self.train_cfg.max_steps_per_epoch > 0
                and step >= self.train_cfg.max_steps_per_epoch
            ):
                break
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=str(self.device), enabled=self.train_cfg.use_amp):
                _, loss = self.model(x, targets=y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if step > 0 and step % self.train_cfg.log_interval == 0:
                avg_loss = total_loss / num_batches
                ppl = math.exp(avg_loss)
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | step {step} | loss {avg_loss:.4f} | ppl {ppl:.2f} "
                    f"| time {elapsed:.0f}s"
                )
                if wandb.run:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/step": step,
                    })

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, epoch: int, step: int):
        """Save model, optimizer and scheduler state for resume."""
        ckpt_dir = os.path.join("outputs", self.model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, "finetune_latest.pt")
        torch.save({
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load checkpoint and return the epoch to resume from."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt["epoch"]

    @torch.no_grad()
    def generate_dialog(self, instruction, input_text="", max_new_tokens=100, temperature=1.0):
        """Generate a response for the given instruction."""
        self.model.eval()
        prompt = build_prompt(instruction, input_text)
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        res_tok = self.model.generate(tokens, max_new_tokens=max_new_tokens, temperature=temperature)
        response = self.tokenizer.decode(res_tok[0][len(tokens[0]):], skip_special_tokens=True)
        self.model.train()
        return response

# ─────────────────────────── Main ───────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = GPTConfig()
    ft_cfg = FinetuneConfig()

    model = GPT(model_cfg).to(device)
    model_name = "gpt-56-5m"
    pretrain_path = os.path.join("outputs", model_name, "final.pt")
    if os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrain_path}")
    else:
        print(f"Warning: no pretrained weights found at {pretrain_path}")

    model_cfg.dropout = ft_cfg.dropout

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:.2f}M")

    # tokenizer
    from pretrain import load_tokenizer
    tokenizer = load_tokenizer(model_cfg.vocab_size)

    # dataset — use same seq len as pretrain so model shapes match
    dataset = AlpacaDataset(tokenizer, model_cfg.max_seq_len, mask_instruction=True)
    dataloader = DataLoader(
        dataset,
        batch_size=ft_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_build_collate_fn(tokenizer.pad_token_id),
    )

    wandb.init(
        project="tiny-gpt-storyteller",
        name=f"finetune_{model_name}",
        config={
            "lr": ft_cfg.learning_rate,
            "epochs": ft_cfg.max_epochs,
            "batch_size": ft_cfg.batch_size,
            "mask_instruction": True,
        },
    )

    finetuner = Finetuner(model, model_cfg, ft_cfg, device, model_name)
    finetuner.tokenizer = tokenizer

    # resume from checkpoint if exists
    start_epoch = 1
    resume_path = os.path.join("outputs", model_name, "finetune_latest.pt")
    if os.path.exists(resume_path):
        start_epoch = finetuner.load_checkpoint(resume_path) + 1
        print(f"Resumed from {resume_path}, starting epoch {start_epoch}")

    for epoch in range(start_epoch, ft_cfg.max_epochs + 1):
        avg_loss = finetuner.train_epoch(dataloader, epoch)
        ppl = math.exp(avg_loss)
        print(f"\n=== Epoch {epoch} done: avg loss {avg_loss:.4f}, ppl {ppl:.2f} ===\n")
        if wandb.run:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch_ppl": ppl, "epoch": epoch})

        # Generate sample responses after each epoch
        tasks = [
            "Write a story about a bear.",
            "Which one is correct? A: The sun makes the sky bright. B: The sun makes the sky black.",
            "Answer the question directly: If a triangle has sides of length 3, 4, and 5, is the angle opposite to the side of length 5 acute, right, or obtuse? Explain why using the Pythagorean theorem.",
        ]
        for task in tasks:
            response = finetuner.generate_dialog(task, max_new_tokens=80, temperature=0.7)
            print(f"── Epoch {epoch} | {task[:60]} ──\n{response}\n")

        finetuner.save_checkpoint(epoch, step=-1)

    # save final model (weights only, for inference)
    final_path = os.path.join("outputs", model_name, "finetuned.pt")
    torch.save(finetuner.model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
