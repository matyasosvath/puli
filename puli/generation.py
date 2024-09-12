from __future__ import annotations

from typing import Callable, List, Union

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
        profile: bool = False,
        compile: bool = False,
        seed: int = 42,
    ) -> Puli:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        start_time = time.time()

        tokenizer = Tokenizer(tokenizer_path)

        model, model_args = Puli.initialize_model(model_name)

        assert model_path.endswith(".pth") or model_path.endswith(".pt"), "model_path should end with '.pt' or '.pth'"

        model.load_state_dict(torch.load(f=model_path, map_location=device, weights_only=True))

        if compile:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        print(f"Model created and loaded in {time.time() - start_time:.2f} seconds from {model_path}")
        print(f"Model has {model.get_num_params()/1e6}M parameters.")

        return Puli(model, tokenizer, model_args, device, profile)

    @staticmethod
    def initialize_model(model_name: str):

        if model_name == "puli2-gpt":

            model_args = puli2_gpt.ModelArgs()
            return puli2_gpt.Puli2GPT(model_args), model_args

        elif model_name == "puli3-gpt-neox":

            model_args = puli3_gpt_neox.ModelArgs()
            return puli3_gpt_neox.Puli3GptNeox(model_args), model_args

        else:
            raise ValueError(f"Model unrecognised! Got {model_name}.")

    def __init__(self, model, tokenizer, model_args, device: torch.device, use_profile: bool = True) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.device = device

        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
            record_shapes=True,
            with_stack=True,
        ) if use_profile else None

    def text_completion(
        self,
        prompt: Union[str, List[str]],
        strategy: str,
        batch_size: int,
        temperature: float = 0.6,
        max_new_tokens: int = 20,
        profile: bool = False,
        **decode_kwargs
    ) -> str:

        max_new_tokens = (
            self.model_args.context_length - 1
            if max_new_tokens > self.model_args.context_length
            else max_new_tokens
        )

        input_tokens = self.tokenizer.encode(prompt, bos=False, eos=False).to(self.device)

        if profile and self.prof: self.prof.start()

        generation_tokens = self.generate(
            input_tokens, max_new_tokens, temperature, strategy, **decode_kwargs
        )

        if profile and self.prof: self.prof.stop()

        return self.tokenizer.decode(generation_tokens.squeeze(0).tolist())

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor, # (batch_size, num_tokens)
        max_new_tokens: int,
        temperature: float,
        strategy: str,
        **decode_kwargs
    ) -> torch.Tensor:

        decode_strategy = self.get_generation_strategy(strategy)

        for _ in range(max_new_tokens):

            # crop current context
            idx_cond = (
                inputs
                if inputs.size(1) <= self.model_args.context_length
                else inputs[:, -self.model_args.context_length :]
            )

            logits = self.model.forward(idx_cond)

            # last time step, (batch, n_token, vocab_size) -> (batch, vocab_size), temperature scaling
            logits = logits[:, -1, :] / temperature

            idx_next = decode_strategy(logits, **decode_kwargs)

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

    def calculate_token_per_second(self, inputs: torch.Tensor, n: int = 100) -> float:

        start_time = time.time()
        self.generate(inputs, n, temperature=1.0, strategy="greedy_decode")
        end_time = time.time()

        total_time = end_time - start_time

        return n / total_time



