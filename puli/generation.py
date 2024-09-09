from __future__ import annotations

from typing import Callable, List, Optional, Union

import time
import torch

from puli.models import puli2_gpt, puli3_gpt_neox
from puli.tokenizer import Tokenizer


class Puli:

    @staticmethod
    def build(
        model_name: str,
        model_path: str,
        tokenizer_path: str,
        device: torch.device,
        seed: int = 42,
    ) -> Puli:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        start_time = time.time()

        tokenizer = Tokenizer(tokenizer_path, device)

        model, model_args = Puli.initialize_model(model_name)

        assert model_path.endswith(".pth") or model_path.endswith(".pt"), "model_path should end with '.pt' or '.pth'"

        model.load_state_dict(torch.load(f=model_path, map_location=device, weights_only=True))

        print(f"Model created and loaded in {time.time() - start_time:.2f} seconds from {model_path}")
        print(f"Model has {model.get_num_params()/1e6}M parameters.")

        return Puli(model_name, model, tokenizer, model_args, device)

    @staticmethod
    def initialize_model(model_name: str):

        if model_name == "puli2-gpt":
            model_args = puli2_gpt.ModelArgs()
            model = puli2_gpt.Puli2GPT(model_args)
            return model, model_args

        elif model_name == "puli3-gpt-neox":
            model_args = puli3_gpt_neox.ModelArgs()
            model = puli3_gpt_neox.Puli3GptNeox(model_args)
            return model, model_args

        else:
            raise ValueError(f"Model unrecognised! Got {model_name}.")

    def __init__(self, model_name: str, model, tokenizer: Tokenizer, model_args, device: torch.device) -> None:

        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.device = device


    def setup(self, batch_size: int, tokens: torch.Tensor, max_new_tokens: int) -> None:

        seq_len = tokens.size(-1) + max_new_tokens

        if self.model_name == "puli3-gpt-neox":
            self.model.setup_caches(batch_size, seq_len, self.device)

        self.profile = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
            record_shapes=True,
            with_stack=True,
        )

    def text_completion(
        self,
        prompt: Union[str, List[str]],
        strategy: str,
        batch_size: int,
        temperature: float = 0.6,
        max_new_tokens: int = 20,
        profile: bool = True,
        **decode_kwargs
    ) -> str:

        max_new_tokens = (
            self.model_args.context_length - 1
            if max_new_tokens > self.model_args.context_length
            else max_new_tokens
        )

        input_tokens, _ = self.tokenizer.encode(prompt, bos=False, eos=False)

        self.setup(batch_size, input_tokens, max_new_tokens)

        if profile: self.profile.start()

        generation_tokens = self.generate(
            input_tokens, None, max_new_tokens, temperature, strategy, **decode_kwargs
        )

        if profile: self.profile.stop()

        return self.tokenizer.decode(generation_tokens.squeeze(0).tolist())

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor, # (batch_size, num_tokens)
        attn_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        strategy: str,
        **decode_kwargs
    ) -> torch.Tensor:

        generation_strategy = self.get_generation_strategy(strategy)

        input_pos = torch.arange(0, inputs.size(-1) + max_new_tokens).to(self.device)

        for _ in range(max_new_tokens):

            _input_pos = input_pos[:inputs.size(-1)]

            # crop current context
            idx_cond = (
                inputs
                if inputs.size(1) <= self.model_args.context_length
                else inputs[:, -self.model_args.context_length :]
            )

            logits = self.model(idx_cond, _input_pos, attn_mask)

            # last time step, (batch, n_token, vocab_size) -> (batch, vocab_size), temperature scaling
            logits = logits[:, -1, :] / temperature

            idx_next = generation_strategy(logits, **decode_kwargs)

            # stop at end-of-sequence token
            if idx_next == self.tokenizer.eos_id: break

            inputs = torch.cat((inputs, idx_next), dim=1)

        return inputs

    def get_generation_strategy(self, strategy: str) -> Callable[..., torch.Tensor]:

        strategies = {
            "greedy_decode": self.greedy_decode,
            "multinomial_sampling": self.multinomial_sampling,
            "top_k_sampling": self.top_k_sampling,
            "top_p_sampling": self.top_p_sampling,
            "beam_decode": None
        }

        if strategy not in strategies:
            raise ValueError(
                f"Strategy {strategy} is not implemented or does not exist! Availables strategies: {strategies}."
            )

        return strategies[strategy]

    def greedy_decode(self, probs: torch.Tensor) -> torch.Tensor:

        # get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

        return idx_next

    def multinomial_sampling(self, logits: torch.Tensor) -> torch.Tensor:

        # probability of each token in vocabulary
        probs = torch.softmax(logits, dim=-1)

        # get the idx of the vocab entry by multinomial sampling
        idx_next = torch.multinomial(probs, num_samples=1)

        return idx_next

    def top_k_sampling(self, logits: torch.Tensor, top_k: int = 3) -> torch.Tensor:

        top_logits, top_pos = torch.topk(logits, top_k)

        # select top k possible tokens, assign -inf to all others in batch
        logits = torch.where(
            condition=logits < top_logits[:, -1],
            input=torch.tensor(float('-inf')),
            other=logits
        )

        # probability of each token in vocabulary
        probs = torch.softmax(logits, dim=-1)

        # get the idx of the vocab entry by multinomial sampling
        idx_next = torch.multinomial(probs, num_samples=1)

        return idx_next

    def top_p_sampling(self, logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:

        assert 0.0 < top_p < 1.0, "top_p must be between 0 and 1."

        # probability of each token in vocabulary
        probs = torch.softmax(logits, dim=-1)

        # sort probabilities in descending order
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # create cumulative sum of elements
        probs_sum = torch.cumsum(probs_sort, dim=-1)

        # mark tokens having values over top_p
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # renormalize remaining probabilities
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # get the idx of the probabilites by multinomial sampling
        idx_next = torch.multinomial(probs_sort, num_samples=1)

        # get original index
        idx_next = torch.gather(probs_idx, -1, idx_next)

        return idx_next
