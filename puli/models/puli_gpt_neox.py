from typing import Tuple

import math
from dataclasses import dataclass

import torch
from torch import nn


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


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):

    ndim = x.ndim

    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_emb(xqkv: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:

    print(f"xqkv shape: {xqkv.shape}")
    print(f"freq_cis shape: {freqs_cis.shape}")

    xq, xk, xv = torch.split(xqkv, 1, dim=2)

    print(f"q shape: {xq.shape}")
    print(f"k shape: {xk.shape}")
    print(f"v shape: {xv.shape}")

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    xqkv = torch.stack([xq_out.type_as(xq), xk_out.type_as(xk), xv], dim=2)

    return xqkv

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device)  # type: ignore

    freqs = torch.outer(t, freqs).float()  # type: ignore

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


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
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):

        batch_size, num_tokens, d_model = x.shape

        # (b, num_tokens, d_model) --> (b, num_tokens, 3 * d_model)
        qkv = self.query_key_value(x)

        # (b, num_tokens, 3 * d_model) --> (b, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)

        qkv = apply_rotary_emb(qkv, freqs_cis)

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

        context_vec = self.dense(context_vec)

        return context_vec


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
        self.freqs_cis = self.freqs_cis.to(x.device)
        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.final_layer_norm(x)
        logits = self.embed_out(x)
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())




if __name__ == "__main__":

    x = torch.tensor([1,2,3,4])
    print(x)
    x = x.unsqueeze(0)
    print(x)

    args = ModelArgs()
    model = Puli3GptNeox(args)

    print(model)
    print("number of parameters: %.2fB" % (model.get_num_params()/1e9,))

    out = model(x)
    print(out)