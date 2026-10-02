"""Mean-pool frozen antigen embeddings to one token per record."""

import csv
import json
from pathlib import Path

import torch


def build_pooled_embeddings(
    records_csv,
    source_dir,
    source_eval,
    output_dir,
    marker,
    tag,
    source_tag,
    splits,
):
    wanted = set(splits)
    with open(records_csv, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in wanted]
    if not rows:
        raise RuntimeError(f"no records found for splits {sorted(wanted)}")

    with open(source_eval) as handle:
        evaluation = json.load(handle)
    if evaluation.get("embedder") != source_tag:
        raise ValueError(
            f"source evaluation is for {evaluation.get('embedder')!r}, "
            f"not {source_tag!r}"
        )

    source_root = Path(source_dir)
    output_root = Path(output_dir)
    built = 0
    reused = 0
    model_name = None
    for index, row in enumerate(rows, start=1):
        record_id = row["id"]
        split = row["split"]
        source = source_root / split / f"{record_id}.pt"
        destination = output_root / split / f"{record_id}.pt"
        if not source.is_file():
            raise FileNotFoundError(f"missing pooled-control source embedding: {source}")

        payload = torch.load(source, map_location="cpu", weights_only=False)
        matrix = payload.get("embedding")
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError(f"invalid source embedding in {source}")
        if payload.get("id") != record_id or payload.get("embedder") != source_tag:
            raise ValueError(f"source provenance mismatch in {source}")
        current_model_name = f"mean_pooled_{payload.get('model_name', source_tag)}_v1"
        if model_name is None:
            model_name = current_model_name
        elif model_name != current_model_name:
            raise ValueError(f"inconsistent source model in {source}")
        pooled = matrix.to(torch.float32).mean(dim=0, keepdim=True)

        valid_cached = False
        if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
            try:
                cached = torch.load(destination, map_location="cpu", weights_only=False)
                valid_cached = (
                    cached.get("id") == record_id
                    and cached.get("embedder") == tag
                    and cached.get("source_embedder") == source_tag
                    and cached.get("model_name") == model_name
                    and cached.get("shape") == list(pooled.shape)
                    and cached.get("source_length") == int(matrix.shape[0])
                )
            except Exception:
                valid_cached = False

        if valid_cached:
            reused += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "embedding": pooled.contiguous(),
                "shape": list(pooled.shape),
                "embedder": tag,
                "model_name": model_name,
                "id": record_id,
                "split": split,
                "source_embedder": source_tag,
                "source_model_name": payload.get("model_name"),
                "source_length": int(matrix.shape[0]),
                "pooling": "mean_over_antigen_tokens_v1",
                "conditioning_scope": "correct_antigen_mean_pooled_v1",
                "target_antigen_used": True,
                "length": 1,
                "dim": int(pooled.shape[1]),
                "chains": payload.get("chains", []),
                "chain_separator_token": False,
            }
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(result, temporary)
            temporary.replace(destination)
            built += 1

        if index % 100 == 0 or index == len(rows):
            print(
                f"[{tag}] {index}/{len(rows)} records "
                f"({built} built, {reused} reused)",
                flush=True,
            )

    result = {
        "status": "complete",
        "tag": tag,
        "method": "antigen_pooled",
        "model_name": model_name,
        "source_embedder": source_tag,
        "source_eval": str(source_eval),
        "records": len(rows),
        "built": built,
        "reused": reused,
        "splits": list(splits),
        "pooling": "mean_over_antigen_tokens_v1",
        "output_tokens": 1,
    }
    marker_path = Path(marker)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[{tag}] pooled-antigen batch complete -> {marker}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("pool_antigen_embeddings.py must run through Snakemake")
    build_pooled_embeddings(
        records_csv=smk.input.records,
        source_dir=smk.params.source_dir,
        source_eval=smk.input.source_eval,
        output_dir=smk.params.output_dir,
        marker=smk.output.marker,
        tag=smk.wildcards.embedder,
        source_tag=smk.params.source_tag,
        splits=list(smk.params.splits),
    )


if __name__ == "__main__":
    main()
