# Tiny GPT Storyteller

从零实现并训练了一个 56.5M 参数的 GPT 模型，完成预训练和指令微调全流程。

## 关键结果

| 阶段 | 数据集 | Loss | PPL |
|------|--------|------|-----|
| 预训练 | TinyStories | **1.37** | **3.93** |
| 指令微调 | Alpaca 52K | **3.52** | **33.76** |

## 训练方法

- **模型**: GPT（d_model=384, 12 头, 10 层, 56.5M 参数），参考 minGPT
- **优化器**: AdamW + Cosine warmup + AMP 混合精度
- **微调策略**: response-only loss masking

## 项目结构

```
├── src/          # 训练代码
└── report/       # 技术报告（NeurIPS 格式）
```
