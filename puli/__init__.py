from typing import Optional, Tuple, Union

import os
import torch
import zipfile
import urllib.request
from tqdm import tqdm

from .generation import Puli


_MODELS = {
    "puli2-gpt": "https://nc.nlp.nytud.hu/s/cCqHLJaftNnRmGZ/download/puli2-gpt.pt",
    "puli3-gpt-neox": ""
}

_TOKENIZERS = {
    "puli2-gpt": "https://nc.nlp.nytud.hu/s/gtnCjHZ2idBZnfb/download/puli2-gpt-tokenizer.zip",
    "puli3-gpt-neox": ""
}


def load_model(
    model_name: str,
    device: torch.device,
    artifact_path: Union[str, None] = None
) -> Puli:

    model_path, tokenizer_dir = _download_artifact(model_name, artifact_path, device)

    puli = Puli.build(model_name, model_path, tokenizer_dir, device)

    puli.model.to(device)

    return puli


def _download_artifact(
    model_name: str,
    artifact_path: Union[str, None] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[str, str]:

    if model_name not in _MODELS or model_name not in _TOKENIZERS:
        raise RuntimeError(f"Model or tokenizer {model_name} not found; available models: {_MODELS}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if artifact_path is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        artifact_path = os.path.join(os.getenv("XDG_CACHE_HOME", default), f"puli/{model_name}")

    model_url = _MODELS[model_name]
    tokenizer_url = _TOKENIZERS[model_name]

    if os.path.isdir(artifact_path):
        print(f"Model path for {model_name} already exists! Skipping download.")
    else:
        _download(model_url, artifact_path)
        _download(tokenizer_url, artifact_path, unzip=True)

    toknizer_dir = artifact_path
    model_file_path = artifact_path + f"/{model_name}.pt"

    return model_file_path, toknizer_dir


def _download(url: str, target_dir: str, unzip: bool = False) -> None:

    os.makedirs(target_dir, exist_ok=True)

    file_name = os.path.basename(url)
    download_path = os.path.join(target_dir, file_name)

    with urllib.request.urlopen(url) as source, open(download_path, "wb") as output:
        with tqdm(
            desc="Downloading data",
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

    if unzip:
        _unzip_file(download_path, target_dir)
        os.remove(download_path)


def _unzip_file(download_path: str, target_dir: str) -> None:

    with zipfile.ZipFile(download_path, "r") as zip_ref:

        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(desc="Unzipping data", total=total_files, unit='file', ncols=80) as progress_bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, target_dir)
                progress_bar.update(1)