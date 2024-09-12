from typing import Optional

import time
from pathlib import Path

import torch

import utils
from puli import generation


def quantize(
    model_name: str,
    checkpoint_path: Path,
    mode: str = 'int8',
    precision: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None
) -> None:

    assert checkpoint_path.is_file(), checkpoint_path

    print(f"Loading {model_name} model from {checkpoint_path}")
    start_time = time.time()

    model, _ = generation.Puli.initialize_model(model_name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)

    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device=device)

    if mode == 'int8':

        model_dynamic_quantized = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )

        print("Dynamic Quantizing for model weights for int8 post-training.")

        dir_name = checkpoint_path.parent
        base_name = checkpoint_path.name
        new_base_name = base_name.replace('.pt', f'.int8.pt')

    else:
        raise ValueError(f"Invalid quantization mode. {mode}, available quantization methods: int8")

    print(f"Writing quantized weights to {dir_name / new_base_name}")
    utils.save_model(model_dynamic_quantized, new_base_name, str(dir_name))

    print(f"Quantization complete took {time.time() - start_time:.02f} seconds")
