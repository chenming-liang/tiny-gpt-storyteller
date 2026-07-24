"""
Evaluation: quantitative (WikiText-2 PPL) + qualitative (generation samples).
Usage: python src/evaluate.py
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

import sys

from config import GPTConfig, TrainConfig, FinetuneConfig
from model import GPT, count_parameters


# ─────────────────────── Data ───────────────────────

def load_wikitext2(tokenizer, max_seq_len, batch_size):
    """Load and tokenize WikiText-2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def tokenize_fn(batch):
        texts = batch["text"]
        # filter out empty lines
        texts = [t for t in texts if t.strip()]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": encodings["input_ids"]}

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # flatten the list of token lists into individual samples
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": nn.utils.rnn.pad_sequence(
                [torch.tensor(b["input_ids"]) for b in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
        },
    )
    return loader, dataset


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
        # loss is mean over non-ignored tokens; multiply by num tokens to get sum
        n_tokens = input_ids.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


# ─────────────────────── Qualitative ───────────────────────

@torch.no_grad()
def generate_samples(model, tokenizer, device, model_name="pretrained"):
    """Generate text from categorized prompts for comprehensive evaluation."""
    # ── 测试分组 ──
    test_cases = {
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

    model.eval()
    print("\n" + "=" * 60)
    print(f"Qualitative Evaluation: {model_name}")
    print("=" * 60)

    for group, prompts in test_cases.items():
        print(f"\n─── {group} ───")
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            output = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n  [Prompt] {prompt}")
            print(f"  [Output] {generated}")
            print()


@torch.no_grad()
def compare_models(pretrained_model, finetuned_model, tokenizer, device):
    """Compare pretrained vs finetuned on the same prompts."""
    comparison_prompts = [
        ("Story prompt", "Once upon a time there was a little"),
        ("Instruction",  "Write a story about a bear."),
    ]

    print("\n" + "=" * 60)
    print("Comparison: Pretrained vs Finetuned")
    print("=" * 60)

    for category, prompt in comparison_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        print(f"\n─── {category}: \"{prompt}\" ───")

        for model, name in [(pretrained_model, "Pretrained"),
                            (finetuned_model, "Finetuned")]:
            output = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n  [{name}]")
            print(f"  {generated}")
            print()


# ─────────────────────── Main ───────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrained"

    model_cfg = GPTConfig()
    train_cfg = TrainConfig()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "pretrained":
        model = GPT(model_cfg).to(device)
        print(f"Model parameters: {count_parameters(model):.2f}M")
        state = torch.load("outputs/pretrained/final.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        print("Checkpoint loaded: pretrained")

        ppl, _ = evaluate_ppl(model, load_wikitext2(
            tokenizer, model_cfg.max_seq_len, train_cfg.batch_size
        )[0], device)
        print(f"\nWikiText-2 → PPL: {ppl:.2f}")

        generate_samples(model, tokenizer, device, "Pretrained")

    elif mode == "finetuned":
        ft_cfg = FinetuneConfig()
        model = GPT(model_cfg).to(device)
        print(f"Model parameters: {count_parameters(model):.2f}M")
        state = torch.load("outputs/gpt-56-5m/finetune_latest.pt", map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print("Checkpoint loaded: finetuned")

        generate_samples(model, tokenizer, device, "Finetuned")

    elif mode == "compare":
        pretrained = GPT(model_cfg).to(device)
        pretrained.load_state_dict(torch.load(
            "outputs/pretrained/final.pt", map_location=device, weights_only=True
        ))

        ft_cfg = FinetuneConfig()
        finetuned = GPT(model_cfg).to(device)
        state = torch.load(
            "outputs/gpt-56-5m/finetune_latest.pt", map_location=device, weights_only=True
        )
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        finetuned.load_state_dict(state)

        generate_samples(pretrained, tokenizer, device, "Pretrained")
        generate_samples(finetuned, tokenizer, device, "Finetuned")
        compare_models(pretrained, finetuned, tokenizer, device)

    else:
        print(f"Usage: python src/evaluate.py [pretrained|finetuned|compare]")
        print("  (default: pretrained)")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
