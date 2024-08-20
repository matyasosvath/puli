from typing import Any, Optional, Union

import io
import os
import torch
import urllib.request
from tqdm import tqdm

from .generation import Puli2
from .model import ModelArgs, PuliGPT
from .tokenizer import Tokenizer


_MODEL = "TODO"


def _download(url: str, root: str) -> Any:

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

    return torch.load(f=download_target)


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[Union[str, None]] = None
) -> PuliGPT:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "puli2")

    if name == _MODEL:
        model_state_dict = _download(_MODEL, download_root)
    elif os.path.isfile(name):
        model_state_dict = torch.load(download_root)
    else:
        raise RuntimeError(f"Model {name} not found; available model: {_MODEL}")

    dims = ModelArgs()
    model = PuliGPT(dims)
    model.load_state_dict(model_state_dict)

    return model.to(device)


def from_pretrained():
    from transformers import AutoModelForCausalLM
    print("Loading weights from pretrained puli-gpt2.")
    model_hf = AutoModelForCausalLM.from_pretrained("NYTK/PULI-GPT-2")
    model_state = model_hf.state_dict()
    model_keys = model_state.keys()


