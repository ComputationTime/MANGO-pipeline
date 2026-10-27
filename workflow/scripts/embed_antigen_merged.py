"""Compress cached, correctly paired ESM2 antigen tokens by cosine similarity.

Merging happens after the frozen ESM2 forward pass. Each round matches even
tokens to odd tokens and merges the strongest edges, carrying token masses so
every original token contributes equally. Groups keep their earliest source
position. No antibody fields or target labels participate.
"""

import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec


MERGING = "cosine_bipartite_weighted_v1"


def _spec_sha256(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def merged_model_name(spec):
    if spec.get("merging") != MERGING:
        raise ValueError(f"unsupported token merging algorithm: {spec.get('merging')!r}")
    fraction = Fraction(str(spec["length_fraction"]))
    if not 0 < fraction <= 1:
        raise ValueError("merged length_fraction must satisfy 0 < fraction <= 1")
    return (f"{spec['source_model_name']}_{MERGING}_"
            f"f{fraction.numerator}of{fraction.denominator}")


def merged_source_path(row, spec):
    return Path(spec["source_dir"]) / row["split"] / f"{row['id']}.pt"


@torch.no_grad()
def merge_tokens(matrix, target_tokens):
    """Return [K,D] centroids and [K,1] masses, deterministically on CPU."""
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 2
            or min(matrix.shape) < 1 or not torch.isfinite(matrix).all()):
        raise ValueError("token merging requires a finite, nonempty [L,D] tensor")
    if not isinstance(target_tokens, int) or not 1 <= target_tokens <= len(matrix):
        raise ValueError("target token count must be an integer between 1 and L")
    values = matrix.detach().to(device="cpu", dtype=torch.float32).clone()
    masses = torch.ones((len(values), 1), dtype=torch.float32)
    # This endpoint is exactly the existing mean-pooled ESM2 representation.
    if target_tokens == 1:
        return values.mean(dim=0, keepdim=True), masses.sum(dim=0, keepdim=True)
    positions = torch.arange(len(values))
    while len(values) > target_tokens:
        remove = min(len(values) - target_tokens, len(values) // 2)
        src, dst = values[::2], values[1::2]
        src_mass, dst_mass = masses[::2], masses[1::2].clone()
        src_pos, dst_pos = positions[::2], positions[1::2].clone()
        scores = F.normalize(src, dim=-1) @ F.normalize(dst, dim=-1).T
        best_score, best_dst = scores.max(dim=-1)
        # Stable ordering makes ties (including zero chain-break vectors) safe.
        ranked = best_score.argsort(descending=True, stable=True)
        selected, retained = ranked[:remove], ranked[remove:]
        destinations = best_dst[selected]
        dst_sum = dst * dst_mass
        dst_sum.index_add_(0, destinations, src[selected] * src_mass[selected])
        dst_mass.index_add_(0, destinations, src_mass[selected])
        dst_pos.scatter_reduce_(0, destinations, src_pos[selected], reduce="amin")
        values = torch.cat((src[retained], dst_sum / dst_mass))
        masses = torch.cat((src_mass[retained], dst_mass))
        positions = torch.cat((src_pos[retained], dst_pos))
        order = positions.argsort(stable=True)
        values, masses, positions = values[order], masses[order], positions[order]
    return values.contiguous(), masses.contiguous()


def antigen_merged(records_csv, record_id, out, seq_source, tag, spec, row=None):
    model_name = merged_model_name(spec)
    row = row if row is not None else ec.load_row(records_csv, record_id)
    if row.get("id") != record_id:
        raise ValueError("merged antigen row id does not match requested record")
    source = merged_source_path(row, spec)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    matrix = payload.get("embedding")
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 2
            or min(matrix.shape) < 1 or payload.get("shape") != list(matrix.shape)):
        raise ValueError(f"invalid token-merging source embedding: {source}")
    if (payload.get("id") != record_id
            or payload.get("embedder") != spec["source_embedder"]
            or payload.get("model_name") != spec["source_model_name"]):
        raise ValueError(f"token-merging source provenance mismatch: {source}")
    if matrix.shape[1] != int(spec["width"]):
        raise ValueError(f"token-merging source width mismatch: {source}")
    fraction = Fraction(str(spec["length_fraction"]))
    tokens = max(1, (len(matrix) * fraction.numerator + fraction.denominator - 1)
                 // fraction.denominator)
    merged, masses = merge_tokens(matrix, tokens)
    meta = ec.build_meta(
        embedder=tag, model_name=model_name, matrix=merged,
        chains=payload.get("chains", []), chain_separator_token=False,
        id=record_id, split=row["split"], seq_source="frozen_antigen_embedding",
        conditioning_scope="correct_antigen_token_merged_v1",
        source_embedder=spec["source_embedder"],
        source_model_name=spec["source_model_name"], source_path=str(source),
        source_length=len(matrix), source_antigen_values_used=True,
        source_chain_separator_token=payload.get("chain_separator_token", False),
        length_fraction=float(fraction), length_rounding="ceil_min_one",
        merging=MERGING, token_masses=masses.squeeze(-1).tolist(),
        merge_spec_sha256=_spec_sha256(spec),
    )
    # Interrupted batches can reuse complete records without accepting partials.
    destination = Path(out)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    ec.save_embedding(str(temporary), merged, meta)
    temporary.replace(destination)


def build_merged_embeddings(records_csv, output_dir, marker, tag, spec, splits):
    from embed_batch import _can_reuse

    with open(records_csv, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in set(splits)]
    if not rows:
        raise ValueError("no records found for token-merging splits")
    model_name = merged_model_name(spec)
    built = reused = 0
    for index, row in enumerate(rows, start=1):
        destination = Path(output_dir) / row["split"] / f"{row['id']}.pt"
        source = merged_source_path(row, spec)
        valid = _can_reuse(destination, row, tag, "antigen_merged", model_name,
                           [source, records_csv, __file__])
        if valid:
            cached = torch.load(destination, map_location="cpu", weights_only=False)
            valid = cached.get("merge_spec_sha256") == _spec_sha256(spec)
        if valid:
            reused += 1
        else:
            antigen_merged(records_csv, row["id"], str(destination),
                           "frozen_antigen_embedding", tag, spec, row=row)
            built += 1
        if index % 100 == 0 or index == len(rows):
            print(f"[{tag}] {index}/{len(rows)} ({built} built, {reused} reused)", flush=True)
    result = dict(status="complete", method="antigen_merged", tag=tag,
                  model_name=model_name, records=len(rows), built=built,
                  reused=reused, splits=list(splits), merging=MERGING,
                  merge_spec_sha256=_spec_sha256(spec))
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("embed_antigen_merged.py must run through Snakemake")
    if hasattr(smk.wildcards, "instance"):
        antigen_merged(smk.input.records, smk.wildcards.instance, smk.output[0],
                       "frozen_antigen_embedding", smk.wildcards.embedder,
                       dict(smk.params.spec))
    else:
        build_merged_embeddings(smk.input.records, smk.params.output_dir,
                                smk.output.marker, smk.wildcards.embedder,
                                dict(smk.params.spec), list(smk.params.splits))


if __name__ == "__main__":
    main()
