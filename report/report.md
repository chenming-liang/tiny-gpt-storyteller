
# Train Your Own Language Models

## 1. Introduction

本项目为课程项目，旨在通过预训练算法和指令微调（SFT）训练一个小型 GPT 模型，以理解训练语言模型的完整流程。

模型采用 Tinystory 数据集进行预训练（https://huggingface.co/datasets/roneneldan/TinyStories），Alpaca 数据进行指令微调。最终预训练模型在 Tinystory 验证集上达到了 1.48 Loss 和 4.41 PPL 且能完成基本的故事续写任务。经过微调的模型能完成对话任务，且在故事写作任务上表现良好。

## 2. Pretraining

### 2.1 Data

项目采用 Tinystory 数据集对模型进行预训练。Tinystory 是一个由 GPT-3.5 和 GPT-4 生成的短篇故事合成数据集，其中的词汇仅限于典型的 3 至 4 岁儿童通常能理解的范围。（Eldan, R., & Li, Y. (2023, 五月 12). _TinyStories: How Small Can Language Models Be and Still Speak Coherent English?_ arXiv.Org. [https://arxiv.org/abs/2305.07759v2](https://arxiv.org/abs/2305.07759v2)）

（TinyStories 数据集描述、数据量、文本长度分布）

### 2.2 Model

项目采用基于 Transformer 的 GPT 架构进行预训练。模型结构如下

- 中间维度：$d_{\text{model}}=384$。
- 多头注意力：$\text{n\_heads}=12$。
- Tranformer 块数量：$10$。
- Tokenizer：采用 AutoTokenizer 库提供的 GPT-2 tokenizer，截断 token 长度设置为 $256$。

（d_model=256, n_layers=6, n_heads=8, max_seq_len=256 → ~53M params）

### 2.3 Training

模型采用 SGD 算法进行训练，训练设置如下。

- 超参数
	- 学习率：$3\times 10^{-4}$。
	- 批次大小：$64$。
	- 训练轮次：$5$。
	- 每轮训练步数：$5000$（由于使用流式数据加载，该值代表每轮训练 5000 次 batch，因此训练步数不受数据集大小限制）。
	- Dropout：$0.0$（未使用 Dropout）
- 优化器：AdamW，$\text{weight\_decay}=0.1,\beta_{1}=0.9,\beta_{2}=0.95$。
- Scheduler：get_cosine_schedule_with_warmup，warmup 步数为 $1000$。
- 损失函数：使用标准交叉熵损失（nn.CrossEntropy）

（lr=3e-4, batch=64, epoch=5, AdamW, cosine schedule）

### 2.4 Results

模型在各个轮次的最终 Loss 和最终 PPL 如下表。

| Epoch | Loss            | PPL             |
| ----- | --------------- | --------------- |
| $1$   | $2.73$          | $15.27$         |
| $2$   | $1.72$          | $5.58$          |
| $3$   | $1.55$          | $4.69$          |
| $4$   | $1.44$          | $4.20$          |
| $5$   | $\mathbf{1.37}$ | $\mathbf{3.93}$ |
Loss 曲线和 PPL 曲线如下图。

![[pretrain_loss.png]]

可以看到经过 $5$ 轮训练之后，模型 Loss 与 PPL 均已已经收敛，未见过拟合现象。

## 3. Instruction Tuning

### 3.1 Data

本项目使用斯坦福大学发布的 Alpaca 数据（Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2023). _Self-Instruct: Aligning Language Models with Self-Generated Instructions_ (arXiv:2212.10560). arXiv. [https://doi.org/10.48550/arXiv.2212.10560](https://doi.org/10.48550/arXiv.2212.10560)）的清洗版本（（https://huggingface.co/datasets/yahma/alpaca-cleaned）进行指令微调训练。Alpaca 数据集包含包含 52K 条指令数据。在本项目中，这些指令数据均被结构化为如下格式：

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{Response}
```

（Alpaca 52K，mask_instruction=True）

### 3.2 Training

指令微调仍然采用 SGD 算法，训练设置如下：

- 超参数
	- 学习率：$3\times 10^{-5}$。
	- 批次大小：$32$。
	- 训练轮次：$5$。
	- 每轮训练步数：$1625$。
- 优化器：AdamW，$\text{weight\_decay}=0.1,\beta_{1}=0.9,\beta_{2}=0.95$。
- Scheduler：get_cosine_schedule_with_warmup，warmup 步数为 $1000$。
- 损失函数：采用标准交叉熵损失（nn.CrossEntropy），并且只对`{response}`部分的内容计算交叉熵损失。

### 3.3 Results

模型在各个轮次的平均 Loss 和平均 PPL 如下表。

| Epoch | Loss            | PPL            |
| ----- | --------------- | -------------- |
| $1$   | $5.26$          | $192.84$       |
| $2$   | $4.17$          | $64.84$        |
| $3$   | $3.81$          | $45.37$        |
| $4$   | $3.62$          | $37.51$        |
| $5$   | $\mathbf{3.52}$ | $\mathbf{33.76}$ |

Loss 曲线和 PPL 曲线如下图。

![[finetune_loss.png]]



## 4. Evaluation

### 4.1 Quantitative

（TinyStories PPL 4.41 / WikiText-2 PPL 10779，表）

### 4.2 Qualitative

（选 3-4 个代表性 prompt 的结果 + 对比分析，表/文字）

## 5. Conclusion