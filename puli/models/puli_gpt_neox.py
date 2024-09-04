from typing import Tuple

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor


@dataclass
class ModelArgs:
    context_length: int = 2048
    vocab_size: int = 50048
    dropout: float = 0.0
    d_model: int = 4096
    eps: float = 1e-05
    n_heads: int = 32
    n_layers: int = 32
    qkv_bias: bool = True


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
    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.dense_h_to_4h = nn.Linear(d_model, 4 * d_model, bias=True)
        self.dense_4h_to_h = nn.Linear(4 * d_model, d_model, bias=True)
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_h_to_4h(x)
        x = self.act(x)
        x = self.dense_4h_to_h(x)
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

        self.query_key_value = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.dense = nn.Linear(d_out, d_out)

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):

        batch_size, num_tokens, d_model = x.shape

        # (b, num_tokens, d_model) --> (b, num_tokens, 3 * d_model)
        qkv = self.query_key_value(x)

        q, k, v = torch.split(qkv, self.d_out, dim=-1)

        q = q.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        k = k.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        v = v.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        context_vec = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.dense(context_vec)

        return context_vec


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


class Block(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.input_layernorm = LayerNorm(args.d_model, args.eps)
        self.post_attention_layernorm = LayerNorm(args.d_model, args.eps)
        self.post_attention_dropout = nn.Dropout(args.dropout, inplace=False)
        self.post_mlp_dropout = nn.Dropout(args.dropout, inplace=False)

        self.attention = MultiHeadedAttention(
            d_in=args.d_model,
            d_out=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
            qkv_bias=args.qkv_bias
            )

        self.mlp = MLP(args.d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_dropout(self.attention(self.input_layernorm(x), freqs_cis))
        x = x + self.post_mlp_dropout(self.mlp(self.post_attention_layernorm(x)))
        return x


class Puli3GptNeox(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.args = args

        self.embed_in = nn.Embedding(args.vocab_size, args.d_model)
        self.embed_dropout = nn.Dropout(args.dropout, inplace=False)
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.final_layer_norm = LayerNorm(args.d_model, args.eps)
        self.embed_out = nn.Linear(args.d_model, args.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.args.d_model // self.args.n_heads,
            self.args.context_length
        )

    def forward(self, idx: torch.Tensor):
        x = self.embed_in(idx)
        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.final_layer_norm(x)
        logits = self.embed_out(x)
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())




if __name__ == "__main__":

    x = torch.randint(0, 2048, (1,2048))
    print(x.shape)

    args = ModelArgs()
    model = Puli3GptNeox(args)

    print(model)
    # print("number of parameters: %.2fB" % (model.get_num_params()/1e9,))

    out = model(x)
    print(out)