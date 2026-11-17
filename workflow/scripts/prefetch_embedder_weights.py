"""Download and verify package-managed embedder weights before compute starts."""

import gc
import json
import os
from pathlib import Path

import torch


def _fair_esm(method, model_name):
    import esm

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"fair-esm has no pretrained loader {model_name!r}")
    model, _alphabet = loader()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model, _alphabet
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    files = sorted(
        str(path) for path in checkpoint_dir.glob(f"{model_name}*.pt")
        if path.is_file() and path.stat().st_size > 0
    )
    if not files:
        raise RuntimeError(
            f"{method} loaded but no nonempty checkpoint matching {model_name} "
            f"was found in {checkpoint_dir}"
        )
    return parameters, files, str(checkpoint_dir)


def _esm3(model_name, representation='sequence'):
    from esm.models.esm3 import ESM3

    # Asset prefetch can overlap another embedder's GPU work. Keep model
    # construction and the lazy structure encoder on CPU in this stage.
    model = ESM3.from_pretrained(model_name, device=torch.device('cpu'))
    if representation == 'sequence_structure':
        # Structure tokenization loads a separate pretrained encoder lazily.
        # Verify its weights before the embedding stage starts.
        model.get_structure_encoder()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    files = []
    if cache_root.exists():
        files = [
            str(path) for path in cache_root.rglob("*")
            if path.is_file() and path.stat().st_size > 0
        ]
    if not files:
        raise RuntimeError(
            f"ESM3 loaded but the Hugging Face cache is empty at {cache_root}"
        )
    return parameters, files, str(cache_root)


def _pyrosetta():
    import pyrosetta

    pyrosetta.init("-mute all", silent=True)
    package = Path(pyrosetta.__file__).resolve()
    database = package.parent / "database"
    if not database.exists():
        raise RuntimeError(f"PyRosetta imported but its database is missing: {database}")
    return 0, [str(package), str(database)], str(package.parent)


def prefetch(method, model_name, tag, output, representation='sequence'):
    print(f"prefetching {tag}: {model_name}", flush=True)
    if method in {"esm2", "esmif"}:
        parameters, files, cache_root = _fair_esm(method, model_name)
    elif method == "esm3":
        parameters, files, cache_root = _esm3(model_name, representation)
    elif method == "pyrosetta_pre":
        parameters, files, cache_root = _pyrosetta()
    else:
        raise ValueError(f"no package-managed weight prefetcher for {method!r}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "status": "ready",
        "tag": tag,
        "method": method,
        "model_name": model_name,
        "parameters": parameters,
        "cache_root": cache_root,
        "cached_file_count": len(files),
        "cached_files": files,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")) if method == "esm3" else None,
        "structure_encoder_verified": method == 'esm3' and representation == 'sequence_structure',
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"{tag} assets ready: {parameters:,} parameters, "
        f"{len(files)} cached file(s) -> {output}",
        flush=True,
    )


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("prefetch_embedder_weights.py is intended for Snakemake")
    prefetch(
        method=smk.params.method,
        model_name=smk.params.model_name,
        tag=smk.wildcards.embedder,
        output=smk.output.marker,
        representation=getattr(smk.params, 'representation', 'sequence'),
    )


if __name__ == "__main__":
    main()
