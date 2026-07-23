"""
Pretraining script for GPT on TinyStories.
Usage: python src/pretrain.py

── 需要你手写的部分 ──
1. TinyStoriesDataset.__iter__: causal LM 的 x 和 y 如何错位构造
2. Trainer._configure_optimizers: 权重衰减分组逻辑
3. Trainer.train_epoch: 训练循环核心（forward, backward, optimizer step）
"""
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb

from config import GPTConfig, TrainConfig
from model import GPT, count_parameters


# ─────────────────────────── Data ───────────────────────────

class TinyStoriesDataset(IterableDataset):
    """Iterable-style dataset that tokenizes texts on-the-fly."""

    def __init__(self, split: str, tokenizer, max_seq_len: int):
        super().__init__()
        self.dataset = load_dataset(
            "roneneldan/TinyStories", split=split, streaming=True
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for example in self.dataset:
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_seq_len,
                return_length=False,
            )["input_ids"]

            # TODO: causal LM 需要输入和目标错一位
            # 用 tokens 构造 x（输入）和 y（目标），每个位置预测下一个 token
            # x 和 y 的长度关系是什么？为什么？
            x, y = tokens[:-1], tokens[1:]

            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─────────────────────────── Tokenizer ───────────────────────────

def load_tokenizer(vocab_size: int = 50257):
    """Load GPT-2 tokenizer (use directly, no training needed)."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_collate_fn(pad_token_id):
    """Dynamic padding within batch. 基础设施，直接抄就行。"""
    def collate_fn(batch):
        x_batch, y_batch = zip(*batch)
        x_padded = nn.utils.rnn.pad_sequence(
            x_batch, batch_first=True, padding_value=pad_token_id
        )
        y_padded = nn.utils.rnn.pad_sequence(
            y_batch, batch_first=True, padding_value=-1  # -1 在 cross_entropy 中被忽略
        )
        return x_padded, y_padded
    return collate_fn


# ─────────────────────────── Training ───────────────────────────

class Trainer:
    """Handles the training loop, logging, evaluation, and checkpointing."""

    def __init__(self, model, train_cfg, tokenizer, device, model_name="gpt"):
        self.model = model
        self.cfg = train_cfg
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name  # e.g. "gpt-3m" for versioning
        self.scaler = GradScaler(enabled=train_cfg.use_amp)

        # optimizer with weight decay groups
        self.optimizer = self._configure_optimizers()

        # scheduler
        total_steps = train_cfg.max_epochs * train_cfg.max_steps_per_epoch  # exact estimate
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_cfg.warmup_steps,
            num_training_steps=total_steps,
        )

    def _configure_optimizers(self):
        """
        TODO: 将参数分为"应用权重衰减"和"不应用权重衰减"两组

        规则：
        - Linear 层的 weight → 应用权重衰减
        - 所有 bias → 不应用权重衰减
        - LayerNorm 和 Embedding 的 weight → 不应用权重衰减

        提示：用 named_modules() + named_parameters() 遍历，
             用 isinstance(m, nn.Linear) / nn.LayerNorm / nn.Embedding 判断模块类型
        """
        decay = set()
        no_decay = set()

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if fpn.endswith('bias'):
                    no_decay.add(fpn)
                elif fpn.endswith('weight'):
                    if isinstance(m, nn.Linear):
                        decay.add(fpn)
                    elif isinstance(m, (nn.LayerNorm, nn.Embedding)):
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter = decay & no_decay
        assert not inter, f"params in both sets: {inter}"
        missing = param_dict.keys() - (decay | no_decay)
        assert not missing, f"params not assigned: {missing}"

        return AdamW(
            [
                {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": self.cfg.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            ],
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.95),
        )

    def train_epoch(self, dataloader, epoch: int):
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for step, (x, y) in enumerate(dataloader):
            if self.cfg.max_steps_per_epoch > 0 and step >= self.cfg.max_steps_per_epoch:
                break
            x, y = x.to(self.device), y.to(self.device)

            # TODO: 训练循环核心
            # 1. 清空梯度
            # 2. 混合精度前向传播 (with autocast)
            # 3. 反向传播
            # 4. 梯度裁剪 (max_norm=1.0)
            # 5. 优化器 step 和 scaler 更新
            # 6. LR scheduler step
            self.optimizer.zero_grad()
            with autocast(device_type=str(self.device), enabled=self.cfg.use_amp):
                _, loss = self.model(x, targets=y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # logging
            if step > 0 and step % self.cfg.log_interval == 0:
                avg_loss = total_loss / num_batches
                ppl = math.exp(avg_loss)
                lr_now = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | step {step} | loss {avg_loss:.4f} | ppl {ppl:.2f} "
                    f"| lr {lr_now:.2e} | time {elapsed:.0f}s"
                )
                if wandb.run:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/lr": lr_now,
                        "train/step": step,
                    })

            # checkpoint
            if step > 0 and step % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, step)

            # generation sample
            if step > 0 and step % self.cfg.sample_interval == 0:
                self.generate_sample(step)

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def generate_sample(self, step: int):
        """Generate text from a prompt for qualitative inspection."""
        self.model.eval()
        for prompt_text in self.cfg.sample_prompts:
            input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.device)
            output = self.model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n── Prompt: {prompt_text} ──")
            print(generated)
            print()
        self.model.train()

    def save_checkpoint(self, epoch: int, step: int, is_final: bool = False):
        """Save model and optimizer state. Only keeps latest + epoch-end checkpoints."""
        ckpt_dir = os.path.join("outputs", self.model_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        # always save latest for resume (single file, overwritten each time)
        path = os.path.join(ckpt_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")


# ─────────────────────────── Main ───────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = GPTConfig()
    train_cfg = TrainConfig()

    model = GPT(model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:.2f}M")

    # auto-version outputs: "outputs/gpt-56M/"
    model_name = "gpt-56-5m"
    print(f"Model version: {model_name}")

    # tokenizer — train a small vocab BPE on TinyStories
    tokenizer = load_tokenizer(model_cfg.vocab_size)

    # datasets
    train_dataset = TinyStoriesDataset("train", tokenizer, model_cfg.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=_build_collate_fn(tokenizer.pad_token_id),
        num_workers=0,     # streaming dataset: 不支持多进程
        pin_memory=True,
    )

    # wandb
    wandb.init(
        project="tiny-gpt-storyteller",
        name=f"pretrain_{model_name}",
        config={
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "vocab_size": model_cfg.vocab_size,
            "max_seq_len": model_cfg.max_seq_len,
            "batch_size": train_cfg.batch_size,
            "lr": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "epochs": train_cfg.max_epochs,
            "warmup_steps": train_cfg.warmup_steps,
        },
    )

    trainer = Trainer(model, train_cfg, tokenizer, device, model_name)

    # resume from checkpoint if exists
    start_epoch = 1
    resume_path = os.path.join("outputs", model_name, "latest.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_path}, starting epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg.max_epochs + 1):
        avg_loss = trainer.train_epoch(train_loader, epoch)
        ppl = math.exp(avg_loss)
        print(f"\n=== Epoch {epoch} done: avg loss {avg_loss:.4f}, ppl {ppl:.2f} ===\n")
        if wandb.run:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch_ppl": ppl, "epoch": epoch})
        trainer.save_checkpoint(epoch, step=-1, is_final=(epoch == train_cfg.max_epochs))

    final_dir = os.path.join("outputs", model_name)
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "final.pt"))
    print(f"Final model saved to {final_dir}/final.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
