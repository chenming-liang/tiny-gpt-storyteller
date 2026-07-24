"""
Evaluation: quantitative (WikiText-2 PPL) + qualitative (generation samples).
Usage:
    python src/evaluate.py pretrained    # PPL + pretrained generation
    python src/evaluate.py finetuned     # finetuned generation
    python src/evaluate.py compare       # both + side-by-side comparison
Output: outputs/evaluation_results.md (appended each run)
"""
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import sys

from config import GPTConfig, TrainConfig, FinetuneConfig
from model import GPT, count_parameters


# ─────────────────────── Helpers ───────────────────────

class Report:
    """Collect evaluation results and write to markdown."""

    def __init__(self):
        self.lines = []

    def add(self, text=""):
        self.lines.append(text)

    def add_code(self, text):
        self.lines.append(f"```\n{text}\n```")

    def save(self, path="outputs/evaluation_results.md"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if os.path.exists(path) else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n\n")
        print(f"\nResults appended to {path}")

    def section(self, title, level=2):
        self.add(f"{'#' * level} {title}")
        self.add()

    def table(self, headers, rows):
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        self.add("| " + " | ".join(headers) + " |")
        self.add(sep)
        for row in rows:
            self.add("| " + " | ".join(str(c) for c in row) + " |")
        self.add()


# ─────────────────────── Data ───────────────────────

def load_wikitext2(tokenizer, max_seq_len, batch_size):
    """Load and tokenize WikiText-2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def tokenize_fn(batch):
        texts = [t for t in batch["text"] if t.strip()]
        encodings = tokenizer(
            texts, truncation=True, max_length=max_seq_len, padding=False, return_tensors=None,
        )
        return {"input_ids": encodings["input_ids"]}

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": nn.utils.rnn.pad_sequence(
                [torch.tensor(b["input_ids"]) for b in batch],
                batch_first=True, padding_value=tokenizer.pad_token_id,
            )
        },
    )
    return loader


# ─────────────────────── Quantitative ───────────────────────

@torch.no_grad()
def evaluate_ppl(model, loader, device):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        logits, loss = model(input_ids, targets=input_ids)
        n_tokens = input_ids.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


# ─────────────────────── Qualitative ───────────────────────

TEST_CASES = {
    "Story Completion": [
        "Once upon a time",
        "The little cat",
        "In a faraway land, there lived a",
        "Tom was walking through the forest when he",
    ],
    "Instruction: Write a Story": [
        "Write a story about a bear.",
        "Write a story about a rabbit that eats a carrot.",
        "Write a story about a duck that swims in a pond and feels happy.",
    ],
    "Instruction: Q&A": [
        "What is the color of the sky?",
        "Why do birds fly south in winter?",
        "What should you do if it rains?",
        "Name three things you can see in a park.",
    ],
    "Factual Completion": [
        "The sun rises in the",
        "Water is made of",
        "The capital of France is",
    ],
}

COMPARISON_PROMPTS = [
    ("Story prompt", "Once upon a time there was a little"),
    ("Instruction", "Write a story about a bear."),
]


@torch.no_grad()
def generate_samples(model, tokenizer, device, model_name="pretrained", report=None):
    """Generate text from categorized prompts. Writes to report."""
    model.eval()
    report.section(f"Qualitative Evaluation: {model_name}")

    for group, prompts in TEST_CASES.items():
        report.add(f"**{group}**")
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            output = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            report.add(f"- **Prompt:** {prompt}")
            report.add(f"  *Output:* {generated}")
            report.add()
            # also print to console
            print(f"  [Prompt] {prompt}")
            print(f"  [Output] {generated}")
            print()


@torch.no_grad()
def compare_models(pretrained_model, finetuned_model, tokenizer, device, report=None):
    """Compare pretrained vs finetuned on same prompts. Writes to report."""
    report.section("Comparison: Pretrained vs Finetuned")

    for category, prompt in COMPARISON_PROMPTS:
        report.add(f"**{category}:** \"{prompt}\"")
        report.add()
        report.add("| Model | Output |")
        report.add("|---|---|")

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        for model, name in [(pretrained_model, "Pretrained"), (finetuned_model, "Finetuned")]:
            output = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            report.add(f"| {name} | {generated} |")
            print(f"  [{name}] {generated}")
        report.add()


# ─────────────────────── Main ───────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrained"
    print(f"Using device: {device}")
    print(f"Mode: {mode}")

    report = Report()
    report.section(f"Evaluation Results ({mode})", level=1)
    report.add(f"- **Date:** 2026-07-24")
    report.add(f"- **Model:** GPT (d_model=384, n_layers=10, n_heads=12, ~53M params)")
    report.add(f"- **Device:** {device}")
    report.add()

    model_cfg = GPTConfig()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "pretrained":
        model = GPT(model_cfg).to(device)
        state = torch.load("outputs/gpt-56-5m/final.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        print("Checkpoint loaded: pretrained")
        report.add(f"- **Model:** Pretrained")
        report.add()

        # Quantitative
        report.section("Quantitative: WikiText-2 Perplexity")
        loader = load_wikitext2(tokenizer, model_cfg.max_seq_len, 32)
        ppl, avg_loss = evaluate_ppl(model, loader, device)
        report.add(f"- **Loss:** {avg_loss:.4f}")
        report.add(f"- **Perplexity:** {ppl:.2f}")
        report.add()
        print(f"WikiText-2 → loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

        # Qualitative
        generate_samples(model, tokenizer, device, "Pretrained", report)

    elif mode == "finetuned":
        model = GPT(model_cfg).to(device)
        state = torch.load("outputs/gpt-56-5m/finetune_latest.pt", map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print("Checkpoint loaded: finetuned")
        report.add(f"- **Model:** Finetuned (on Alpaca)")
        report.add()

        generate_samples(model, tokenizer, device, "Finetuned", report)

    elif mode == "compare":
        pretrained = GPT(model_cfg).to(device)
        pretrained.load_state_dict(torch.load(
            "outputs/gpt-56-5m/final.pt", map_location=device, weights_only=True
        ))
        print("Checkpoint loaded: pretrained")

        finetuned = GPT(model_cfg).to(device)
        state = torch.load("outputs/gpt-56-5m/finetune_latest.pt", map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        finetuned.load_state_dict(state)
        print("Checkpoint loaded: finetuned")

        report.add(f"- **Models:** Pretrained & Finetuned")
        report.add()

        generate_samples(pretrained, tokenizer, device, "Pretrained", report)
        generate_samples(finetuned, tokenizer, device, "Finetuned", report)
        compare_models(pretrained, finetuned, tokenizer, device, report)
    else:
        print(f"Usage: python src/evaluate.py [pretrained|finetuned|compare]")
        return

    report.save()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
