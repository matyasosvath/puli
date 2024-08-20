from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import PuliGPT, ModelArgs
from .tokenizer import Tokenizer


class Puli2:

    def __init__(self, model: PuliGPT, tokenizer: Tokenizer, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build(model_path: str, tokenizer_path: str) -> Puli2:

        model_args = ModelArgs()

        model = PuliGPT(model_args)
        tokenizer = Tokenizer(tokenizer_path)

        assert model_path.endswith(".pth") or model_path.endswith(".pt"), "model_path should end with '.pt' or '.pth'"

        print(f"Loading model from {model_path}")

        model.load_state_dict(torch.load(f=model_path))

        return Puli2(model, tokenizer, model_args)

    @torch.no_grad()
    def generate(
        self,
        input: torch.Tensor, # (batch_size, tokens)
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):

            idx_cond = ( # crop sequence len
                input
                if input.size(1) <= self.model_args.context_length
                else input[:, -self.model_args.context_length :]
            )

            logits = self.model.forward(idx_cond)

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