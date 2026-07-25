# Tiny GPT Storyteller

Train a small GPT model from scratch — pretraining on TinyStories and instruction tuning on Alpaca.

## Project Structure

```
├── src/                # Training code
├── report/
│   ├── report.tex      # Technical report (NeurIPS format)
│   ├── pretrain_loss.png
│   └── finetune_loss.png
├── my_notes/           # Personal notes
└── .gitignore
```

## Highlights

- **Model**: GPT (d_model=384, 12 heads, 10 layers, ~56.5M params)
- **Pretrain**: TinyStories dataset — 1.37 Loss, 3.93 PPL
- **Finetune**: Alpaca 52K instruction dataset
- **Report**: Written in NeurIPS 2023 format (see `report/report.tex`)

## Compile Report

```bash
cd report
xelatex report.tex
```

Requires `neurips.sty` (included) and `ctex` LaTeX package.
