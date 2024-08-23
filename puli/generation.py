from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from .model import Puli2GPT, ModelArgs
from .tokenizer import Tokenizer


class Puli2:

    @staticmethod
    def build(model_path: str, tokenizer_path: str, seed: int = 42) -> Puli2:

        torch.manual_seed(seed)

        start_time = time.time()

        model_args = ModelArgs()
        tokenizer = Tokenizer(tokenizer_path)
        model = Puli2GPT(model_args)

        assert model_path.endswith(".pth") or model_path.endswith(".pt"), "model_path should end with '.pt' or '.pth'"

        model.load_state_dict(torch.load(f=model_path, map_location="cpu"))

        print(f"Model created and loaded in {time.time() - start_time:.2f} seconds from {model_path}")
        print(f"Model has {model.get_num_params()/1e6}M parameters.")

        return Puli2(model, tokenizer, model_args)

    def __init__(self, model: Puli2GPT, tokenizer: Tokenizer, model_args: ModelArgs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

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