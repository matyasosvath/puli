from typing import Any, Optional, Union

import io
import os
import torch
import urllib.request
from tqdm import tqdm

from .generation import Puli2
from .model import ModelArgs, PuliGPT
from .tokenizer import Tokenizer


_MODEL = {
    "puli-gpt2": "https://nc.nlp.nytud.hu/s/p26z5Yzc3mAjo6K/download/puli-gpt2.pt"
}


def __download(url: str, root: str) -> Any:

    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def load_model(
    name: str,
    model_path: str,
    tokenizer_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> PuliGPT:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    default = os.path.join(os.path.expanduser("~"), ".cache")
    model_dir = os.path.join(os.getenv("XDG_CACHE_HOME", default), "puli")

    if name in _MODEL:
        model_path = __download(_MODEL[name], model_path)
    elif os.path.isfile(name):
        model_path = torch.load(model_path)
    else:
        raise RuntimeError(f"Model {name} not found; available models: {_MODEL}")

    model = Puli2.build(model_path, tokenizer_path)

    return model.to(device)






