"""Fail-fast CUDA validation executed inside the exact model environment."""

import json
import os
import platform
import subprocess
from pathlib import Path

import torch


def preflight(min_memory_gb, output):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable inside workflow/envs/model.yaml. Confirm that "
            "nvidia-smi works and that the CUDA-enabled Conda environment resolved."
        )
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    memory_gb = props.total_memory / 2**30
    if memory_gb < float(min_memory_gb):
        raise RuntimeError(
            f"GPU has {memory_gb:.1f} GiB; configured minimum is {min_memory_gb} GiB"
        )
    # Force an actual CUDA kernel and synchronize so driver/runtime failures
    # occur here rather than hours into embedding or training.
    left = torch.randn((1024, 1024), device=device)
    right = torch.randn((1024, 1024), device=device)
    checksum = float((left @ right).mean().item())
    torch.cuda.synchronize(device)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        smi = "unavailable"
    result = {
        "status": "ok",
        "hostname": platform.node(),
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_memory_gb": round(memory_gb, 2),
        "driver_version": smi,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kernel_checksum": checksum,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("gpu_preflight.py is intended to run through Snakemake")
    preflight(smk.params.min_memory_gb, smk.output.report)


if __name__ == "__main__":
    main()
