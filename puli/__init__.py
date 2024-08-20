from typing import Any, Optional, Union

import os
import torch
import urllib.request
from tqdm import tqdm

from .generation import Puli2


_MODELS = {
    "puli2-gpt": "https://nc.nlp.nytud.hu/s/p26z5Yzc3mAjo6K/download/puli-gpt2.pt"
}


def load_model(
    model_name: str,
    model_path: Union[str, None] = None,
    tokenizer_path: Union[str, None] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:

    if model_name not in _MODELS:
        raise RuntimeError(f"Model {model_name} not found; available models: {_MODELS}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        model_path = os.path.join(os.getenv("XDG_CACHE_HOME", default), "puli")

    model_url = _MODELS[model_name]

    os.makedirs(model_path, exist_ok=True)
    download_path = os.path.join(model_path, os.path.basename(model_url))

    if os.path.isfile(download_path): print(f"Model path for {model_name} already exists! Skipping download")
    else: _download(model_url, download_path)

    puli = Puli2.build(download_path, tokenizer_path)

    puli.model.to(device)

    return puli


def _download(url: str, download_path: str) -> Any:

    if os.path.exists(download_path) and not os.path.isfile(download_path):
        raise RuntimeError(f"{download_path} exists and is not a regular file")

    with urllib.request.urlopen(url) as source, open(download_path, "wb") as output:
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

    return download_path
