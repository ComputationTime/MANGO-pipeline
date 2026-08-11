"""Consistent CUDA selection and fail-fast behavior for GPU executions."""

import os

import torch


def cuda_required() -> bool:
    return os.environ.get("MANGO_REQUIRE_CUDA", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def get_device(purpose: str = "this job") -> torch.device:
    """Return CUDA when available; refuse silent CPU fallback in GPU mode."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        print(
            f"CUDA {purpose}: {props.name}, capability={props.major}.{props.minor}, "
            f"memory={props.total_memory / 2**30:.1f} GiB, torch_cuda={torch.version.cuda}",
            flush=True,
        )
        return device
    if cuda_required():
        raise RuntimeError(
            f"MANGO_REQUIRE_CUDA=1 but PyTorch cannot access CUDA for {purpose}. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}. "
            "Check the NVIDIA driver, GPU visibility, and CUDA-enabled rule environment."
        )
    return torch.device("cpu")
