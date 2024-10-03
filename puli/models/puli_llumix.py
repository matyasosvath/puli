from __future__ import annotations

from typing import Optional

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelArgs:
    batch_size: int = 32
    context_length: int = 4096 # seq_len, block_size
    vocab_size: int = 32_000
    dropout: float = 0.0
    d_model: int = 4096 # hidden_size
    eps: float = 1e-05
    n_heads: int = 2 # 32
    n_layers: int = 2 # 32
    mlp_bias: bool = False
    intermediate_size: int = 11_008 # ffn_hidden_size
    attn_dropout: float = 0.0
    attn_bias: bool = False
    rope_base: int = 10_000


class RoPE(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, base: int = 10_000) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.seq_len = seq_len
        self.base = base

        self.rope_init()

    def rope_init(self) -> None:

        # inverse of scaled frequencies, rotation factors
        inv_freqs = 1.0 / self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)

        # create position indexes `[0, 1, ..., seq_len - 1]`
        pos_idx = torch.arange(self.seq_len, dtype=inv_freqs.dtype, device=inv_freqs.device)

         # outer product of inverse frequencies and position index
        freqs = torch.einsum("i, j -> ij", pos_idx, inv_freqs).float()

        # create rotary matry by stacking sine and cosine components, used to rotate the embeddings
        rotary_embeddings = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1) # (seq_len, head_dim // 2, 2)

        self.register_buffer("rotary_embeddings", rotary_embeddings, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x := (batch_size, seq_len, n_heads, head_dim)

        # (batch_size, seq_len, n_heads, head_dim // 2, 2)
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # (1, seq_len, 1, head_dim // 2, 2)
        rope_cache = self.rotary_embeddings.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # apply rotary embeddings
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # (batch_size, seq_len, n_heads, head_dim)
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class MultiHeadedAttention(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.d_model % args.n_heads == 0, "d_model is indivisible by n_heads"

        self.args = args

        self.n_heads = self.args.n_heads
        self.head_dim = self.args.d_model // self.args.n_heads
        self.d_model = self.args.d_model
        self.attn_dropout = self.args.attn_dropout

        self.q_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)
        self.k_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)
        self.v_proj = nn.Linear(args.d_model, args.d_model, bias=args.attn_bias)

        self.o_proj = nn.Linear(args.d_model, args.d_model, bias=False)

        self.rotary_emb = RoPE(self.head_dim, args.context_length, args.rope_base)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):

        batch_size, seq_length, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_length, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.n_heads, self.head_dim)

        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        context_vec = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=True
        )

        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )

        context_vec = self.o_proj(context_vec)

        return context_vec


class MLP(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Decoder(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.self_attn = MultiHeadedAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.LayerNorm(args.d_model, args.eps)
        self.post_attention_layernorm = nn.LayerNorm(args.d_model, args.eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class PuliLlumix(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.args = args

        self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([Decoder(args) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.d_model, args.eps)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
