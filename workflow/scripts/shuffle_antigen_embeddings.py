"""Build a deterministic, within-split shuffled-antigen negative control.

Each target receives the frozen ESM2 representation of a different record from
the same split. A Sattolo permutation guarantees no fixed points while retaining
the real ESM2 tensor and antigen-length distributions. The mapping depends only
on record ids, split names, and the configured seed; antibody data are unused.
"""

import csv
import hashlib
import json
import random
from pathlib import Path

import torch


def _split_seed(seed: int, split: str) -> int:
    blob = f"mango-shuffled-antigen-v1\0{seed}\0{split}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def derangement(ids, seed: int, split: str):
    """Return a deterministic Sattolo permutation (one cycle, no fixed point)."""
    donors = list(ids)
    if len(donors) < 2:
        raise ValueError(f"split {split!r} needs at least two records to shuffle")
    rng = random.Random(_split_seed(seed, split))
    for index in range(len(donors) - 1, 0, -1):
        other = rng.randrange(index)
        donors[index], donors[other] = donors[other], donors[index]
    if any(target == donor for target, donor in zip(ids, donors)):
        raise AssertionError(f"internal error: shuffled split {split!r} has a fixed point")
    return donors


def build_shuffled_embeddings(
    records_csv,
    source_dir,
    source_eval,
    output_dir,
    marker,
    mapping_csv,
    tag,
    source_tag,
    seed,
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

    rows_by_split = {split: [] for split in splits}
    for row in rows:
        rows_by_split[row["split"]].append(row)

    assignments = []
    for split in splits:
        split_rows = rows_by_split[split]
        ids = [row["id"] for row in split_rows]
        donors = derangement(ids, int(seed), split)
        assignments.extend(
            (split, target, donor) for target, donor in zip(ids, donors)
        )

    source_root = Path(source_dir)
    output_root = Path(output_dir)
    built = 0
    reused = 0
    model_name = None
    map_rows = []
    for index, (split, target_id, donor_id) in enumerate(assignments, start=1):
        source = source_root / split / f"{donor_id}.pt"
        destination = output_root / split / f"{target_id}.pt"
        if not source.is_file():
            raise FileNotFoundError(f"missing shuffled-control donor embedding: {source}")

        donor = torch.load(source, map_location="cpu", weights_only=False)
        matrix = donor.get("embedding")
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError(f"invalid donor embedding in {source}")
        if donor.get("id") != donor_id or donor.get("embedder") != source_tag:
            raise ValueError(f"donor provenance mismatch in {source}")
        if model_name is None:
            model_name = f"shuffled_{donor.get('model_name', source_tag)}_v1"

        valid_cached = False
        if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
            try:
                cached = torch.load(destination, map_location="cpu", weights_only=False)
                valid_cached = (
                    cached.get("id") == target_id
                    and cached.get("donor_id") == donor_id
                    and cached.get("embedder") == tag
                    and cached.get("source_embedder") == source_tag
                    and cached.get("model_name") == model_name
                    and cached.get("shape") == list(matrix.shape)
                )
            except Exception:
                valid_cached = False

        if valid_cached:
            reused += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "embedding": matrix.detach().to("cpu", torch.float32).contiguous(),
                "shape": list(matrix.shape),
                "embedder": tag,
                "model_name": model_name,
                "id": target_id,
                "split": split,
                "donor_id": donor_id,
                "source_embedder": source_tag,
                "source_model_name": donor.get("model_name"),
                "conditioning_scope": "shuffled_antigen_within_split_v1",
                "target_antigen_used": False,
                "pairing_correct": False,
                "shuffle_seed": int(seed),
                "length": int(matrix.shape[0]),
                "dim": int(matrix.shape[1]),
                "chains": donor.get("chains", []),
                "chain_separator_token": donor.get("chain_separator_token", False),
            }
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)
            built += 1

        map_rows.append({
            "split": split,
            "target_id": target_id,
            "donor_id": donor_id,
            "target_equals_donor": target_id == donor_id,
            "donor_length": int(matrix.shape[0]),
            "embedding_width": int(matrix.shape[1]),
        })
        if index % 100 == 0 or index == len(assignments):
            print(
                f"[{tag}] {index}/{len(assignments)} records "
                f"({built} built, {reused} reused)",
                flush=True,
            )

    mapping_path = Path(mapping_csv)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)

    mapping_sha256 = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    result = {
        "status": "complete",
        "tag": tag,
        "method": "antigen_shuffled",
        "model_name": model_name,
        "source_embedder": source_tag,
        "source_eval": str(source_eval),
        "records": len(assignments),
        "built": built,
        "reused": reused,
        "splits": list(splits),
        "shuffle_seed": int(seed),
        "fixed_points": 0,
        "mapping_csv": str(mapping_csv),
        "mapping_sha256": mapping_sha256,
    }
    marker_path = Path(marker)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[{tag}] shuffled-antigen batch complete -> {marker}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("shuffle_antigen_embeddings.py must run through Snakemake")
    build_shuffled_embeddings(
        records_csv=smk.input.records,
        source_dir=smk.params.source_dir,
        source_eval=smk.input.source_eval,
        output_dir=smk.params.output_dir,
        marker=smk.output.marker,
        mapping_csv=smk.output.mapping,
        tag=smk.wildcards.embedder,
        source_tag=smk.params.source_tag,
        seed=int(smk.params.seed),
        splits=list(smk.params.splits),
    )


if __name__ == "__main__":
    main()
