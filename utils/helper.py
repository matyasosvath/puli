import sys
import thop
import torch


DTYPES = {"float16": 2, "float32": 4, "float64": 8}


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {params / 1_000_000}M parameters.")
    return params


def calculate_model_size(model: torch.nn.Module, dtype: str = "float32") -> float:
    total_params = count_parameters(model)
    total_size_bytes = total_params * DTYPES[dtype]
    total_size_mb = total_size_bytes / (1024 * 1024) # convert to megabytes
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    return total_size_mb


def calculate_flops(input: torch.Tensor, model: torch.nn.Module, device=None) -> None:

    if device: model = model.to(device)

    # MACS (multiply-accumulate operations) ~ 2 FLOPS (1 multiply, 1 accumulate)
    macs, params = thop.profile(model, inputs=(input,), verbose=False)
    flops = 2*macs

    print(f"{params:18}: {flops:.1e} FLOPS")

    del model; torch.cuda.empty_cache()


def check_parallelism_requirements() -> None:
    gpu_num = torch.cuda.device_count()
    if sys.platform == 'win32': print('Windows platform is not supported for pipeline parallelism'); sys.exit(0)
    if gpu_num < 2: print('Need at least two GPU devices for parallelism.'); sys.exit(0)
    else: print(f'You have {gpu_num} GPU devices for parallelism.')


def get_model_bandwidth_utilization(
    model: torch.nn.Module,
    dtype: str,
    tokens_per_second: float,
    gpu_type: str = "A100_80GB"
) -> int:

    gpus_memory_bandwith = {"A100_80GB": 1e12, "A100_40GB": None}

    params = count_parameters(model)
    bytes_per_param = DTYPES[dtype]
    memory_bandwith = gpus_memory_bandwith[gpu_type]

    return (params * bytes_per_param * tokens_per_second) / memory_bandwith

