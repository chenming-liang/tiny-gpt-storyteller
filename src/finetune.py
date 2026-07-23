"""
Instruction finetuning script for GPT on Alpaca dataset.
Usage: python src/finetune.py

── 需要你手写的部分 ──
1. generate_dialog: 微调后与模型对话
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
    """将 Alpaca 数据拼接成完整文本（prompt + response）。"""
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
            # 构造 prompt 部分（不含 output）并算出 token 长度
            prompt_text = build_prompt(example["instruction"], example["input"])
            prompt_len = len(self.tokenizer(prompt_text, truncation=False)["input_ids"])
            # 将 prompt 对应的 y 位置设为 -1
            for i in range(min(prompt_len - 1, len(y))):
                y[i] = -1

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─────────────────────────── Finetuning ───────────────────────────

class Finetuner:
    """Handles finetuning loop, loading pretrained weights, and dialog."""

    def __init__(self, model_cfg, train_cfg, device, model_name="gpt"):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.model_name = model_name

        # load pretrained model
        self.model = GPT(model_cfg).to(device)
        self._load_pretrained()

        n_params = count_parameters(self.model)
        print(f"Model parameters: {n_params:.2f}M")

        # lower LR for finetuning
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,     # should be 1e-5 ~ 5e-5
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        self.scaler = GradScaler(enabled=train_cfg.use_amp)

    def _load_pretrained(self):
        """
        TODO: 加载预训练好的模型权重

        预训练权重路径: outputs/{model_name}/final.pt
        注意: 这个文件只保存了 model.state_dict()，没有 optimizer 状态

        提示:
        - 检查文件是否存在，不存在则警告并跳过
        - 用 torch.load(..., map_location=self.device) 加载
        - 用 self.model.load_state_dict(...) 载入
        """
        # ===== 你的代码从这里开始 =====
        path = os.path.join('outputs', self.model_name, 'final.pt')
        if not os.path.exists(path):
            print(f"Warning: no pretrained weights found at {path}, starting from scratch")
            return
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state)
        # ===== 你的代码到这里结束 =====

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

    def save_checkpoint(self):
        """Save finetuned model weights."""
        ckpt_dir = os.path.join("outputs", self.model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, "finetuned.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Finetuned model saved: {path}")

    @torch.no_grad()
    def generate_dialog(self, instruction, input_text="", max_new_tokens=100):
        """
        TODO: 与微调后的模型对话

        1. 用 build_prompt 构造 prompt（不含 output）
        2. 用 self.tokenizer 编码
        3. 调用 self.model.generate() 生成回复
        4. 解码并打印结果
        """
        self.model.eval()

        # ===== 你的代码从这里开始 =====
        prompt = build_prompt(instruction, input_text)
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        res_tok = self.model.generate(tokens, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(res_tok[0][len(tokens[0]):], skip_special_tokens=True)
        # ===== 你的代码到这里结束 =====

        self.model.train()
        return response

# ─────────────────────────── Main ───────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = GPTConfig()
    ft_cfg = FinetuneConfig()

    # 微调时加一点 dropout 防止过拟合
    model_cfg.dropout = ft_cfg.dropout

    # 模型名称（和预训练时一致，用于加载权重）
    model_name = "gpt-6-9m"

    # tokenizer（需要和预训练时用的是同一个）
    from pretrain import train_tokenizer
    tokenizer = train_tokenizer(model_cfg.vocab_size)

    # dataset
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

    finetuner = Finetuner(model_cfg, ft_cfg, device, model_name)
    # 把 tokenizer 挂到 finetuner 上，供 generate_dialog 使用
    finetuner.tokenizer = tokenizer

    for epoch in range(1, ft_cfg.max_epochs + 1):
        avg_loss = finetuner.train_epoch(dataloader, epoch)
        ppl = math.exp(avg_loss)
        print(f"\n=== Epoch {epoch} done: avg loss {avg_loss:.4f}, ppl {ppl:.2f} ===\n")
        if wandb.run:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch_ppl": ppl, "epoch": epoch})

        # Generate a sample response after each epoch
        response = finetuner.generate_dialog("Write a short story about a friendly dragon.")
        print(f"── Epoch {epoch} Sample ──\n{response}\n")

    finetuner.save_checkpoint()

    wandb.finish()


if __name__ == "__main__":
    main()
