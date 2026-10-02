"""Antigen-free control representations for MANGO.

Both controls emit exactly one 1280-wide token by default, matching the input
width of the ESM2 650M representation without exposing antigen sequence or
length. ``constant`` emits the same zero vector for every record; after the
linear antigen projection, its trainable bias acts as a learned no-antigen
token. ``random`` emits a deterministic Gaussian vector keyed only by the
record id and configured seed. No antibody or antigen-derived value is used.
"""

import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec


MODES = {"constant", "random"}


def control_model_name(spec: dict) -> str:
    mode = str(spec["mode"])
    width = int(spec.get("width", 1280))
    tokens = int(spec.get("tokens", 1))
    seed = int(spec.get("seed", 0))
    return f"antigen_control_{mode}_d{width}_l{tokens}_seed{seed}_v1"


def _record_seed(seed: int, record_id: str) -> int:
    blob = f"mango-antigen-control-v1\0{seed}\0{record_id}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") & ((1 << 63) - 1)


def control_matrix(record_id: str, spec: dict) -> torch.Tensor:
    mode = str(spec["mode"])
    if mode not in MODES:
        raise ValueError(f"control mode must be one of {sorted(MODES)}, got {mode!r}")
    width = int(spec.get("width", 1280))
    tokens = int(spec.get("tokens", 1))
    if width < 1 or tokens < 1:
        raise ValueError("control width and tokens must be positive")
    if mode == "constant":
        return torch.zeros((tokens, width), dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_record_seed(int(spec.get("seed", 0)), record_id))
    return torch.randn((tokens, width), generator=generator, dtype=torch.float32)


def antigen_control(
    records_csv: str,
    record_id: str,
    out: str,
    seq_source: str,
    tag: str,
    spec: dict,
    row: dict | None = None,
) -> None:
    # Loading the row verifies the record exists, but none of its biological
    # columns participate in the representation.
    row = row if row is not None else ec.load_row(records_csv, record_id)
    if row.get("id") != record_id:
        raise ValueError(f"control row id {row.get('id')!r} does not match {record_id!r}")
    matrix = control_matrix(record_id, spec)
    ec.save_embedding(
        out,
        matrix,
        meta=ec.build_meta(
            embedder=tag,
            model_name=control_model_name(spec),
            matrix=matrix,
            chains=[],
            chain_separator_token=False,
            id=record_id,
            seq_source="none",
            conditioning_scope="no_antigen_fixed_token_v1",
            source_antigen_used=False,
            control_mode=str(spec["mode"]),
            control_seed=int(spec.get("seed", 0)),
        ),
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        antigen_control(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            out=smk.output[0],
            seq_source=smk.params.seq_source,
            tag=smk.wildcards.embedder,
            spec=dict(smk.params.spec),
        )
        return

    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--id", required=True, dest="record_id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--spec", required=True, help="JSON embedder specification")
    args = parser.parse_args()
    antigen_control(
        args.records, args.record_id, args.out, "none", args.tag,
        json.loads(args.spec),
    )


if __name__ == "__main__":
    main()
