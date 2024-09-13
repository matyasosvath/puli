from pathlib import Path

import torch


def save_model(model: torch.nn.Module, model_name: str, target_dir: str = "models") -> None:

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"Saving model to: {model_save_path}")

    # torch.save(obj=model.state_dict(), f=model_save_path)
    torch.save(obj=model, f=model_save_path)


def load_model(model: torch.nn.Module, model_path: Path, device: torch.device) -> torch.nn.Module:

    assert model_path.is_file(), f"model_path {model_path} is not a file."
    assert str(model_path).endswith(".pth") or str(model_path).endswith(".pt"), "model_name should end with '.pt' or '.pth'"

    print(f"Loading model from {model_path}")

    model.load_state_dict(torch.load(f=model_path, map_location=device, weights_only=True))

    return model.to(device)


def load_quantized_model(
    model: torch.nn.Module,
    model_path: Path,
    mode: str,
    device: torch.device,
    precision: torch.dtype = torch.bfloat16
) -> torch.nn.Module:

    assert mode in ["int8"], f"Quantization {mode} not available"
    assert str(model_path).endswith(".pth") or str(model_path).endswith(".pt"), "model_name should end with '.pt' or '.pth'"

    if model_path.name.endswith(".pt"): model_path_str = str(model_path).replace(".pt", f".{mode}.pt")
    else: model_path_str = str(model_path).replace(".pth", f".{mode}.pth")

    print(f"Loading {mode} weight-only quantized model from {model_path_str}")

    model = replace_linear_weight_only_int8_per_channel(model)

    checkpoint = torch.load(str(model_path_str), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)

    return model.eval()


def replace_linear_weight_only_int8_per_channel(model: torch.nn.Module):

    for name, child in model.named_children():
        if isinstance(child, torch.nn.Linear) and name == "lm_head":
            setattr(model, name, WeightOnlyInt8Linear(child.in_features, child.out_features, bias=False))
        elif isinstance(child, torch.nn.Linear):
            setattr(model, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
        else:
            replace_linear_weight_only_int8_per_channel(child)

    return model


class WeightOnlyInt8Linear(torch.nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor
    scales: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))
        self.register_buffer("bias", torch.zeros((out_features), dtype=torch.bfloat16)) if bias else None


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(input, self.weight.to(dtype=input.dtype))
        x = x * self.scales
        x = x + self.bias.to(dtype=input.dtype) if self.bias is not None else x
        return x

