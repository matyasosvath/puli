from typing import Tuple

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelArgs:
    context_length: int = 1024
    vocab_size: int = 50048
    dropout: float = 0.1
    d_model: int = 1024
    eps: float = 1e-05
    n_heads: int = 16
    n_layers: int = 24
    qkv_bias: bool = True


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model)) # scale
        self.bias = nn.Parameter(torch.zeros(d_model)) # shift
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * norm_x + self.bias


class GELU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()

        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.act = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:

        super().__init__()

        assert d_out % n_heads == 0, "d_model is indivisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.d_out = d_out

        self.c_attn = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.c_proj = nn.Linear(d_out, d_out)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, d_model = x.shape

        # (b, num_tokens, d_model) --> (b, num_tokens, 3 * d_model)
        qkv = self.c_attn(x)

        # (b, num_tokens, 3 * d_model) --> (b, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)

        # (b, num_tokens, 3, n_heads, head_dim) --> (3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, n_heads, num_tokens, head_dim) -> 3 times (b, n_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )

        # combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.resid_dropout(self.c_proj(context_vec))

        return context_vec


class Block(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.ln_1 = LayerNorm(args.d_model, args.eps)
        self.attn = MultiHeadedAttention(
            d_in=args.d_model,
            d_out=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
            qkv_bias=args.qkv_bias
            )

        self.ln_2 = LayerNorm(args.d_model, args.eps)
        self.mlp = MLP(args.d_model, args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Puli2GPT(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.args = args

        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.wpe = nn.Embedding(args.context_length, args.d_model)
        self.drop= nn.Dropout(args.dropout)
        self.h = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.ln_f = LayerNorm(args.d_model, args.eps)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):

        device = idx.device
        batch_size, seq_len = idx.shape

        tok_embeds = self.wte(idx)
        pos_embeds = self.wpe(torch.arange(seq_len, device=device))
        x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_size)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
