"""
GPT model for TinyStories pretraining (~1.2M params).
Reference: https://github.com/karpathy/minGPT
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_k * self.n_heads, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2) # (B, n_heads, T, d_k)

        # TODO: scaled dot-product attention + output projection
        mul_qk = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        mul_qk = mul_qk.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf")) # (B, n_heads, T, T)
        mh_att = self.dropout(torch.softmax(mul_qk, dim=-1)) @ v # (B, n_heads, T, d_k)
        att = self.c_proj(mh_att.transpose(1, 2).contiguous().view(B, T, C))
        return self.dropout(att)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        # TODO: define two linear layers (d_model -> 4*d_model -> d_model) with GELU
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )

    def forward(self, x):
        # TODO: linear -> gelu -> linear
        return self.fc(x)

class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, config):
        super().__init__()
        # TODO: init layernorm, attention, mlp
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)


    def forward(self, x):
        # TODO: pre-norm residual blocks (attn + mlp)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full GPT model."""

    def __init__(self, config):
        super().__init__()

        # TODO: token embedding + position embedding
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.f_ln = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # (1, T)
        tok = idx
        pos = self.pos_emb(pos) # (1, T, d_model)
        tok = self.tok_emb(tok) # (B, T, d_model)
        x = self.dropout(pos + tok) # (B, T, d_model)

        for b in self.blocks:
            x = b(x)
        logits = self.proj(self.f_ln(x))
        # TODO: embedding lookup -> transformer blocks -> lm_head -> logits

        loss = None
        if targets is not None:
            # TODO: cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def count_parameters(model):
    """Return total number of trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
