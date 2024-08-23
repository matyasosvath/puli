from typing import List

import os
import time
from logging import getLogger
from transformers import AutoTokenizer


logger = getLogger()


class Tokenizer:

    def __init__(self, tokenizer_dir: str) -> None:

        assert os.path.isdir(tokenizer_dir), tokenizer_dir

        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        logger.info(f"Loading tokenizer in {time.time() - start_time:.2f} seconds from {tokenizer_dir}")

        self.vocab_size: int = len(self.tokenizer.get_vocab())

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        logger.info(f"Vocab size: {self.vocab_size} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:

        assert type(text) is str, f"Parameter `text` must be string. Got {type(text)}"

        tokens = self.tokenizer.encode(text)

        if bos: tokens = [self.bos_id] + tokens
        if eos: tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)