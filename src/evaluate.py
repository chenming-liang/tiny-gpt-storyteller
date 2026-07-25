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
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from config import GPTConfig
from model import GPT
from finetune import build_prompt


# ────────────────────────── Configuration ──────────────────────────

CHECKPOINTS = {
    "pretrained": "outputs/gpt-56-5m/final.pt",
    "finetuned":  "outputs/gpt-56-5m/finetuned.pt",
}

MODEL_DESC   = "GPT (d_model=384, n_layers=10, n_heads=12, ~53M params)"

# Generation config per model variant
TEXT_TEMP     = 1.0          # raw text completion
INSTRUCT_TEMP = 0.7          # instruction-following

# ── Test prompts ──

RAW_PROMPTS = {
    "Story Completion": [
        "Once upon a time there was a little girl named Lily. She loved to explore the park near her house. One sunny afternoon, she saw a ladder leaning against a big tree. Curious about what was at the top, she started to climb. But the ladder began to wobble...",
        "Once upon a time there was a little boy named Max. Max loved to help his family in the garden. One day, he dug a deep hole and found a mysterious wooden box covered in dirt. He carefully opened it and saw...",
        "Once upon a time there was a very kind wizard. He lived in a tall tower and spent his days helping the villagers nearby. One morning, he heard a knock on the door and found a tiny baby dragon on his doorstep...",
        "Once upon a time there was a brave little duck named Dottie. Dottie wasn't afraid of the dark, or thunderstorms, or even the big fish who lived in the pond. But one day, Dottie had to cross a very wobbly bridge to get to her nest. As she stepped onto the bridge...",
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
        "Write a short essay about why we need to protect the environment.",
        "Explain how a light bulb works in simple terms a child would understand.",
    ],
}

COMPARE_STORY       = "Once upon a time there was a little girl named Lucy. She was very adventurous. She loved to explore the world around her, especially when it was bright and sunny outside. One day, while exploring the nearby park, Lucy came across a ladder leaning on a wall. She was curious to see what was on top, so she climbed the ladder, but when she reached the top, the ladder fell and she was stuck. A nearby park ranger noticed her and shouted out,"
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

# (evaluate_ppl now works directly on HuggingFace datasets — no loader needed.)


# ────────────────────────── Evaluation ──────────────────────────

@torch.no_grad()
def evaluate_ppl(model, dataset, tokenizer, device, max_seq_len=256):
    """Compute perplexity with correct causal LM evaluation.

    At each position t, the model predicts token t+1.
    This mirrors how loss is computed during training: x=tokens[:-1], y=tokens[1:].
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for example in dataset:
        tokens = tokenizer(
            example["text"], truncation=True, max_length=max_seq_len,
        )["input_ids"]
        if len(tokens) < 2:
            continue
        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(tokens[1:],  dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)                     # forward without loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        total_loss += loss.item() * len(tokens)
        total_tokens += len(tokens)
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


@torch.no_grad()
def generate_one(model, tokenizer, device, prompt, *,
                 temperature=1.0, max_new_tokens=150,
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
    # checkpoint dict has "model_state_dict" key; bare state_dict doesn't
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(f"Loaded {variant} from {path}")
    return model


def run_pretrained(device, tokenizer, report):
    model = _load_model(device, "pretrained")

    # Quantitative — in-domain (TinyStories validation)
    report.section("Quantitative: TinyStories Validation Perplexity")
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    ppl, avg_loss = evaluate_ppl(model, dataset, tokenizer, device,
                                 GPTConfig.max_seq_len)
    report.kv("Loss", f"{avg_loss:.4f}")
    report.kv("Perplexity", f"{ppl:.2f}")
    report.add()
    print(f"TinyStories → loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

    # Quantitative — out-of-domain (WikiText-2)
    report.section("Quantitative: WikiText-2 Perplexity")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ppl, avg_loss = evaluate_ppl(model, dataset, tokenizer, device,
                                 GPTConfig.max_seq_len)
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
