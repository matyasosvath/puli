from typing import List, Tuple, Union

import os
import time
import torch
from transformers import AutoTokenizer


class Tokenizer:

    def __init__(self, tokenizer_dir: str, device: torch.device) -> None:

        assert os.path.isdir(tokenizer_dir), tokenizer_dir

        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        print(f"Loading tokenizer in {time.time() - start_time:.2f} seconds from {tokenizer_dir}")

        self.device = device
        self.vocab_size: int = len(self.tokenizer.get_vocab())

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        print(f"Vocab size: {self.vocab_size} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")


    def encode(
        self, text: Union[str, List[str]], bos: bool = True, eos: bool = True,
    ):

        assert isinstance(text, str) or isinstance(text, list), f"Parameter `text` must be string or list. Got {type(text)}"

        tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        tokens, attention_mask = tokens["input_ids"], tokens["attention_mask"]

        if bos: tokens = torch.tensor([self.bos_id]) + tokens
        if eos: tokens = tokens + torch.tensor([self.eos_id])

        attention_mask = (attention_mask == 1)  # Convert integer mask to bool

        return tokens.to(self.device), attention_mask.to(self.device)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
