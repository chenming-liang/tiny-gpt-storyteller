"""
Evaluation: quantitative (WikiText-2 PPL) + qualitative (generation samples).

Usage:
    python src/evaluate.py pretrained    # PPL + generation
    python src/evaluate.py finetuned     # generation (instruction format)
    python src/evaluate.py compare       # both side-by-side

Output: outputs/evaluation_results.md
"""
import math
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from config import GPTConfig, FinetuneConfig
from model import GPT
from finetune import build_prompt  # noqa: E402


# ────────────────────────── Configuration ──────────────────────────

CHECKPOINTS = {
    "pretrained": "outputs/gpt-56-5m/final.pt",
    "finetuned":  "outputs/gpt-56-5m/finetuned.pt",
}

MODEL_DESC   = "GPT (d_model=384, n_layers=10, n_heads=12, ~53M params)"

# Generation config per model variant
TEXT_TEMP     = 1.0          # raw text completion
INSTRUCT_TEMP = 0.7          # instruction-following
MAX_NEW_TOKENS = 50

# ── Test prompts ──

RAW_PROMPTS = {
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

INSTRUCT_PROMPTS = {
    "Instruction: Write a Story": RAW_PROMPTS["Instruction: Write a Story"],
    "Instruction: Q&A":          RAW_PROMPTS["Instruction: Q&A"],
    "Instruction: Complex": [
        "Explain why we need to save water.",
        "Describe how a bicycle works.",
    ],
}

COMPARE_STORY       = "Once upon a time there was a little"
COMPARE_INSTRUCTION = "Write a story about a bear."


# ────────────────────────── Report ──────────────────────────

class Report:
    """Accumulates markdown and writes to file."""

    def __init__(self):
        self._lines = []

    def add(self, text=""):
        self._lines.append(text)

    def section(self, title, level=2):
        self._lines.append(f"{'#' * level} {title}\n")

    def table(self, headers, rows):
        self._lines.append("| " + " | ".join(headers) + " |")
        self._lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            self._lines.append("| " + " | ".join(str(c) for c in row) + " |")
        self._lines.append("")

    def header(self, title):
        self._lines.append(f"# {title}\n")

    def kv(self, key, value):
        self._lines.append(f"- **{key}:** {value}")

    def save(self, path="outputs/evaluation_results.md"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")
        print(f"\nReport saved to {path}")


# ────────────────────────── Data ──────────────────────────

def load_wikitext2(tokenizer, max_seq_len, batch_size):
    """Load WikiText-2 test set (generic text evaluation).

    Note: for TinyStories-trained models, use load_tinystories_val()
    for in-domain evaluation instead.
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    return _prepare_text_dataset(dataset, tokenizer, max_seq_len, batch_size)


def load_tinystories_val(tokenizer, max_seq_len, batch_size):
    """Load TinyStories validation set (in-domain evaluation)."""
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    return _prepare_text_dataset(dataset, tokenizer, max_seq_len, batch_size)


def _prepare_text_dataset(dataset, tokenizer, max_seq_len, batch_size):
    """Tokenize a text dataset and return a DataLoader."""

    def tokenize_fn(examples):
        texts = examples["text"]
        encodings = tokenizer(
            texts, truncation=True, max_length=max_seq_len,
            padding=False, return_tensors=None,
        )
        return {"input_ids": encodings["input_ids"]}

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
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


# ────────────────────────── Evaluation ──────────────────────────

@torch.no_grad()
def evaluate_ppl(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        _, loss = model(input_ids, targets=input_ids)
        n = input_ids.numel()
        total_loss += loss.item() * n
        total_tokens += n
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


@torch.no_grad()
def generate_one(model, tokenizer, device, prompt, *,
                 temperature=1.0, max_new_tokens=50,
                 format_as_instruction=False):
    """Generate a single response.

    Args:
        format_as_instruction: wrap prompt with Alpaca template.
    Returns:
        generated text (prompt prefix included).
    """
    if format_as_instruction:
        prompt = build_prompt(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens,
                            temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_generation_suite(model, tokenizer, device, model_name,
                         prompt_groups, temperature, report):
    """Run all prompt groups and write to report & stdout."""
    report.section(f"Qualitative Evaluation: {model_name}")

    for group, prompts in prompt_groups.items():
        report.add(f"**{group}**\n")
        for prompt in prompts:
            generated = generate_one(model, tokenizer, device, prompt,
                                     temperature=temperature)
            report.add(f"- **Prompt:** {prompt}")
            report.add(f"  *Output:* {generated}\n")
            print(f"  [{group}] {prompt}")
            print(f"  → {generated}\n")


# ────────────────────────── Main ──────────────────────────

def _load_model(device, variant):
    """Load a model and its checkpoint.

    Args:
        variant: "pretrained" or "finetuned".
    Returns:
        GPT model.
    """
    cfg = GPTConfig()
    model = GPT(cfg).to(device)

    path = CHECKPOINTS[variant]
    state = torch.load(path, map_location=device, weights_only=True)
    # finetuned.pt is bare state_dict (no wrapper); latest.pt is a checkpoint dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        pass  # already bare state_dict
    model.load_state_dict(state)
    print(f"Loaded {variant} from {path}")
    return model


def run_pretrained(device, tokenizer, report):
    model = _load_model(device, "pretrained")

    # Quantitative — in-domain (TinyStories validation)
    report.section("Quantitative: TinyStories Validation Perplexity")
    loader = load_tinystories_val(tokenizer, GPTConfig.max_seq_len, batch_size=32)
    ppl, avg_loss = evaluate_ppl(model, loader, device)
    report.kv("Loss", f"{avg_loss:.4f}")
    report.kv("Perplexity", f"{ppl:.2f}")
    report.add()
    print(f"TinyStories → loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

    # Quantitative — out-of-domain (WikiText-2)
    report.section("Quantitative: WikiText-2 Perplexity")
    loader = load_wikitext2(tokenizer, GPTConfig.max_seq_len, batch_size=32)
    ppl, avg_loss = evaluate_ppl(model, loader, device)
    report.kv("Loss", f"{avg_loss:.4f}")
    report.kv("Perplexity", f"{ppl:.2f}")
    report.add()
    print(f"WikiText-2 → loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

    # Qualitative
    run_generation_suite(model, tokenizer, device, "Pretrained",
                         RAW_PROMPTS, temperature=TEXT_TEMP, report=report)


def run_finetuned(device, tokenizer, report):
    model = _load_model(device, "finetuned")
    run_generation_suite(model, tokenizer, device,
                         "Finetuned (instruction format)",
                         INSTRUCT_PROMPTS, temperature=INSTRUCT_TEMP,
                         report=report)


def run_compare(device, tokenizer, report):
    pretrained = _load_model(device, "pretrained")
    finetuned  = _load_model(device, "finetuned")

    # ── preprint runs ──
    run_generation_suite(pretrained, tokenizer, device, "Pretrained",
                         RAW_PROMPTS, temperature=TEXT_TEMP, report=report)
    run_generation_suite(finetuned, tokenizer, device,
                         "Finetuned (instruction format)",
                         INSTRUCT_PROMPTS, temperature=INSTRUCT_TEMP,
                         report=report)

    # ── side-by-side comparison ──
    report.section("Comparison: Pretrained vs Finetuned")

    # Story completion: both raw text
    report.add(f"**Story completion:** \"{COMPARE_STORY}\"\n")
    report.table(["Model", "Output"], [
        ["Pretrained", generate_one(pretrained, tokenizer, device,
                                     COMPARE_STORY, temperature=TEXT_TEMP)],
        ["Finetuned",  generate_one(finetuned,  tokenizer, device,
                                     COMPARE_STORY, temperature=TEXT_TEMP)],
    ])

    # Instruction: pretrained raw, finetuned Alpaca
    report.add(f"**Instruction:** \"{COMPARE_INSTRUCTION}\"\n")
    report.table(["Model", "Output"], [
        ["Pretrained (raw text)",
         generate_one(pretrained, tokenizer, device,
                      COMPARE_INSTRUCTION, temperature=TEXT_TEMP)],
        ["Finetuned (Alpaca format)",
         generate_one(finetuned,  tokenizer, device,
                      COMPARE_INSTRUCTION, temperature=INSTRUCT_TEMP,
                      format_as_instruction=True)],
    ])
    print("Comparison done.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrained"
    print(f"Device: {device}  |  Mode: {mode}")

    report = Report()
    report.header(f"Evaluation Results ({mode})")
    report.kv("Date", "2026-07-24")
    report.kv("Model", MODEL_DESC)
    report.kv("Device", str(device))
    report.add()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "pretrained":
        run_pretrained(device, tokenizer, report)
    elif mode == "finetuned":
        run_finetuned(device, tokenizer, report)
    elif mode == "compare":
        run_compare(device, tokenizer, report)
    else:
        print(f"Usage: python src/evaluate.py [pretrained|finetuned|compare]")
        return

    report.save()


if __name__ == "__main__":
    main()
