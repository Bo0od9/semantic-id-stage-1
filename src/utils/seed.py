from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, cudnn_deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ``cudnn_deterministic=False`` даёт 10–30% пропускной способности на CUDA
    # за счёт автотюна ядер; для research-репорта с 3 seeds воспроизводимости
    # достаточно torch/numpy/random seeds.
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = not cudnn_deterministic


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_device_memory(device: torch.device | str) -> None:
    """Drop cached allocations for MPS/CUDA — important on MPS between large evals."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
