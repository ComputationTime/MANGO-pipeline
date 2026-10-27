"""Constant and random control representations for MANGO.

The configured controls match each record's full frozen ESM2 tensor shape,
retaining prefix length while replacing every value with zeros (``constant``)
or deterministic Gaussian noise (``random``). Noise depends only on record id,
seed and shape. No antibody input or source embedding values are used.

Fractional controls retain ceil(length_fraction * L), with a one-token minimum.
The fixed-token layout remains available with shape_mode="fixed".
"""

import hashlib
from fractions import Fraction
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec


MODES = {"constant", "random"}


def length_fraction(spec: dict) -> Fraction:
    fraction = Fraction(str(spec.get("length_fraction", 1)))
    if not 0 < fraction <= 1:
        raise ValueError("control length_fraction must satisfy 0 < fraction <= 1")
    if spec.get("shape_mode", "fixed") != "match_source" and fraction != 1:
        raise ValueError("length_fraction requires shape_mode='match_source'")
    return fraction


def control_model_name(spec: dict) -> str:
    mode = str(spec["mode"])
    width = int(spec.get("width", 1280))
    seed = int(spec.get("seed", 0))
    fraction = length_fraction(spec)
    if spec.get("shape_mode", "fixed") == "match_source":
        if fraction != 1:
            return (
                f"antigen_control_{mode}_d{width}_shape_like_{spec['source_embedder']}_"
                f"f{fraction.numerator}of{fraction.denominator}_seed{seed}_v3"
            )
        return (
            f"antigen_control_{mode}_d{width}_shape_like_"
            f"{spec['source_embedder']}_seed{seed}_v2"
        )
    tokens = int(spec.get("tokens", 1))
    return f"antigen_control_{mode}_d{width}_l{tokens}_seed{seed}_v1"


def control_source_path(row: dict, spec: dict) -> Path | None:
    shape_mode = spec.get("shape_mode", "fixed")
    if shape_mode == "fixed":
        return None
    if shape_mode != "match_source":
        raise ValueError(f"unknown control shape_mode {shape_mode!r}")
    return Path(spec["source_dir"]) / row["split"] / f"{row['id']}.pt"


def _source_shape(path: Path, record_id: str, spec: dict) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"missing control shape reference: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    matrix = payload.get("embedding")
    if (
        not isinstance(matrix, torch.Tensor)
        or matrix.ndim != 2
        or min(matrix.shape) < 1
        or payload.get("shape") != list(matrix.shape)
    ):
        raise ValueError(f"invalid control shape reference: {path}")
    if (
        payload.get("id") != record_id
        or payload.get("embedder") != spec["source_embedder"]
        or payload.get("model_name") != spec["source_model_name"]
    ):
        raise ValueError(f"control shape reference provenance mismatch: {path}")
    return tuple(matrix.shape)


def _record_seed(seed: int, record_id: str) -> int:
    blob = f"mango-antigen-control-v1\0{seed}\0{record_id}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") & ((1 << 63) - 1)


def control_matrix(record_id: str, spec: dict, *, shape=None) -> torch.Tensor:
    mode = str(spec["mode"])
    if mode not in MODES:
        raise ValueError(f"control mode must be one of {sorted(MODES)}, got {mode!r}")
    width = int(spec.get("width", 1280))
    fraction = length_fraction(spec)
    shape_mode = spec.get("shape_mode", "fixed")
    if shape_mode == "match_source":
        if shape is None or len(shape) != 2:
            raise ValueError("match_source controls require a 2-D reference shape")
        source_tokens, source_width = map(int, shape)
        if source_width != width:
            raise ValueError(
                f"control width {width} does not match source width {source_width}"
            )
        if source_tokens < 1:
            raise ValueError("control source length must be positive")
        tokens = max(1, (source_tokens * fraction.numerator + fraction.denominator - 1)
                     // fraction.denominator)
    elif shape_mode == "fixed":
        tokens = int(spec.get("tokens", 1))
        source_tokens = tokens
    else:
        raise ValueError(f"unknown control shape_mode {shape_mode!r}")
    if width < 1 or tokens < 1:
        raise ValueError("control width and tokens must be positive")
    if mode == "constant":
        return torch.zeros((tokens, width), dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_record_seed(int(spec.get("seed", 0)), record_id))
    # Draw the full reference tensor before truncation so fractional contexts
    # are exact nested prefixes, independent of RNG kernel shape choices.
    noise = torch.randn((source_tokens, width), generator=generator, dtype=torch.float32)
    return noise[:tokens].clone() if tokens < source_tokens else noise


def antigen_control(
    records_csv: str,
    record_id: str,
    out: str,
    seq_source: str,
    tag: str,
    spec: dict,
    row: dict | None = None,
) -> None:
    # Only record identity and source tensor shape participate. Antigen sequence
    # columns and antibody columns do not determine the output values.
    row = row if row is not None else ec.load_row(records_csv, record_id)
    if row.get("id") != record_id:
        raise ValueError(f"control row id {row.get('id')!r} does not match {record_id!r}")
    source = control_source_path(row, spec)
    shape = _source_shape(source, record_id, spec) if source is not None else None
    matrix = control_matrix(record_id, spec, shape=shape)
    fraction = length_fraction(spec)
    scope = "no_antigen_fixed_token_v1"
    if source is not None:
        scope = ("no_antigen_values_source_shape_v1" if fraction == 1
                 else "no_antigen_values_source_length_fraction_v1")
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
            seq_source="shape_only" if source is not None else "none",
            conditioning_scope=scope,
            source_antigen_used=source is not None,
            source_antigen_shape_used=source is not None,
            source_antigen_values_used=False,
            shape_reference=str(source) if source is not None else None,
            source_length=shape[0] if shape is not None else None,
            length_fraction=float(fraction) if source is not None else None,
            length_rounding="ceil_min_one" if source is not None else None,
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
