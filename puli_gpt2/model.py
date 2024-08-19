from typing import Tuple

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from puli_gpt2 import config


@dataclass
class ModelArgs:
    batch_size: int = 32
    context_length: int = 1024
    vocab_size: int = 50_257
    betas: Tuple[float, float] = (0.9, 0.98)
    dropout: float = 0.1
    d_model: int = 512
    eps: float = 1e-09
    d_ff: int = 2048
    lr: float = 3e-4
    lr_warmup: int = 16_000
    n_heads: int = 8
    n_layers: int =  6
    qkv_bias: bool = False


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

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:

        super().__init__()

        assert d_out % n_heads == 0, "d_model is indivisible by n_heads"

        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_out // n_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, d_model = x.shape

        # (b, num_tokens, d_model) --> (b, num_tokens, 3 * d_model)
        qkv = self.qkv(x)

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

        context_vec = self.proj(context_vec)

        return context_vec


class DecoderBlock(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.attn = MultiHeadedAttention(
            d_in=config.d_model,
            d_out=config.d_model,
            context_length=config.context_length,
            n_heads=config.n_heads,
            dropout=config.dropout,
            qkv_bias=config.qkv_bias
            )

        self.norm1 = LayerNorm(config.d_model, config.eps)
        self.norm2 = LayerNorm(config.d_model, config.eps)
        self.ff = FeedForward(config.d_model, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Decoder(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x) -> torch.Tensor:
        return self.blocks(x)


class GPTModel(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.config = config

        self.tok_emb = Embeddings(config.vocab_size, config.d_model)
        self.pos_emb = Embeddings(config.context_length, config.d_model)
        self.dropout= nn.Dropout(config.dropout)
        self.decoder = Decoder(config)
        self.norm = LayerNorm(config.d_model, config.eps)
        self.generator = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.tok_emb.weight = self.generator.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)

        print(f"Number of parameters: {self.get_num_params()/1e6}M")

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor):

        device = idx.device
        batch_size, seq_len = idx.shape

        tok_embeds = self.tok_emb(idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=device))
        x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_size)
        x = self.dropout(x)
        x = self.decoder(x)
        x = self.norm(x)
        logits = self.generator(x)
        return logits

    def _init_weights(self, module) -> None:

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(
        self,
        input: torch.Tensor, # (batch_size, tokens)
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_k: int = config.TOP_K,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):

            idx_cond = ( # crop sequence len
                input
                if input.size(1) <= self.config.context_length
                else input[:, -self.config.context_length :]
            )

            logits = self.forward(idx_cond)

            # temperature scaling
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

             # decoding strategies, multinomial sampling
            idx_next = torch.multinomial(probs, num_samples=1)

            input = torch.cat((input, idx_next), dim=1)

        return input
