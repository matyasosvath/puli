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

        # model_dynamic_quantized = torch.quantization.quantize_dynamic(
        #     model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        # )
        model_dynamic_quantized = create_quantized_state_dict(model)

        print("Dynamic Quantizing for model weights for int8 post-training.")

        dir_name = checkpoint_path.parent
        base_name = checkpoint_path.name
        new_base_name = base_name.replace('.pt', f'.int8.pt')

    else:
        raise ValueError(f"Invalid quantization mode. {mode}, available quantization methods: int8")

    # input_ids = ids_tensor([8, 1024], 50_048)
    # traced_model = torch.jit.trace(model_dynamic_quantized, (input_ids))
    # torch.jit.save(traced_model, "test_quant.pt")

    print(f"Writing quantized weights to {dir_name / new_base_name}")
    utils.save_model(model_dynamic_quantized, new_base_name, str(dir_name))

    print(f"Quantization complete took {time.time() - start_time:.02f} seconds")


@torch.no_grad()
def create_quantized_state_dict(model: torch.nn.Module):
    cur_state_dict = model.state_dict()
    for fqn, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            int8_weight, scales, _ = dynamically_quantize_per_channel(mod.weight.float(), -128, 127, torch.int8)
            cur_state_dict[f"{fqn}.weight"] = int8_weight
            cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)

    return cur_state_dict

def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


# def ids_tensor(size = (8,1024), vocab_size: int = 50_000):
#     #  Creates a random int32 tensor of the shape within the vocab size
#     return torch.randint(0, vocab_size, size, dtype=torch.bfloat16, device='cpu')


if __name__ == "__main__":

    m = "puli2-gpt"
    p = Path("/home/osvathm/.cache/puli/puli2-gpt/puli2-gpt.pt")

    quantize(m,p)