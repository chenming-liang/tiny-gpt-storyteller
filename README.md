# Tiny GPT Storyteller

从零训练一个小型 GPT 模型——在 TinyStories 上预训练，在 Alpaca 上指令微调。

## 项目结构

```
├── src/                # 训练代码
├── report/
│   ├── report.tex      # 技术报告（NeurIPS 格式）
│   ├── pretrain_loss.png
│   └── finetune_loss.png
└── .gitignore
```

## 关键结果

- **模型**: GPT（d_model=384, 12 头注意力, 10 层, ~56.5M 参数）
- **预训练**: TinyStories 数据集 — Loss 1.37, PPL 3.93
- **微调**: Alpaca 52K 指令数据集
