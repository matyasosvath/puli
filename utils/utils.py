from pathlib import Path

import torch


def save_model(model: torch.nn.Module, model_name: str, target_dir: str = "models", log: bool = True) -> None:

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"Saving model to: {model_save_path}") if log else None

    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model: torch.nn.Module, model_name: str, target_dir: str = "models") -> torch.nn.Module:

    target_dir_path = Path(target_dir)

    model_path = target_dir_path / model_name
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

    print(f"Loading model from {model_path}")

    model.load_state_dict(torch.load(f=model_path))

    return model