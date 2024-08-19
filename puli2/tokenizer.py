from typing import List
import os

from sentencepiece import SentencePieceProcessor

from . import config


class Tokenizer:

    def __init__(self, model_file: str = config.TOKENISER_MODEL_PATH) -> None:

        assert os.path.isfile(model_file), model_file

        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.Load(model_file)

        print(f"Loaded SentencePiece model from {model_file}")

        self.n_words: int = self.tokenizer.vocab_size()

        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()

        print(f"Words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:

        assert type(text) is str

        tokens = self.tokenizer.Encode(text)

        if bos: tokens = [self.bos_id] + tokens
        if eos: tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.Decode(tokens)