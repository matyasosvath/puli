from typing import Any, Optional, Tuple, Union

import os
import torch
import zipfile
import urllib.request
from tqdm import tqdm

from .generation import Puli


_ARTIFACTS = {
    "puli2-gpt": "https://nc.nlp.nytud.hu/s/RBwpYYF6XCNdaWy/download/puli2-gpt.zip",
    "puli3-gpt-neox": "TODO"
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
    model_path: Union[str, None] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[str, str]:

    if model_name not in _ARTIFACTS:
        raise RuntimeError(f"Model {model_name} not found; available models: {_ARTIFACTS}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        model_path = os.path.join(os.getenv("XDG_CACHE_HOME", default), f"puli/{model_name}")

    artifact_url = _ARTIFACTS[model_name]

    if os.path.isdir(model_path): print(f"Artifact path for {model_name} already exists! Skipping download.")
    else: _download(artifact_url, model_path)

    toknizer_dir = model_path
    model_file_path = model_path + "/model.pt"

    return model_file_path, toknizer_dir


def _download(url: str, target_dir: str) -> None:

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

    with zipfile.ZipFile(download_path, "r") as zip_ref:

        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(desc="Unzipping data", total=total_files, unit='file', ncols=80) as progress_bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, target_dir)
                progress_bar.update(1)

    os.remove(download_path)
