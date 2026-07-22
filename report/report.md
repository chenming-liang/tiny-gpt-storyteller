租用 autodl RTX(S)4080 的卡，GPU 数量 1，cuda>=11.8

pytorch 2.8+python 3.12+cuda 12.8

在 hugging face 上加载 tinystory 数据集，使用 gpt2 tokenizer，pad token=eos token，用于补全末尾

激活函数 GeLU