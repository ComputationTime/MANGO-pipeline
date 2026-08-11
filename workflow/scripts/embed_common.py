"""Shared helpers for the per-structure embedding scripts.

Each embedding script processes ONE record (one ``.pt`` output), reading the
sequence(s) it needs from ``records.csv`` so sequence-based methods never
re-parse structures and always agree with the resolved/expected choice made at
processing time.

Output layout (the embedder contract)
-------------------------------------
    <emb_dir>/<tag>/embedder_config.json     one per embedder
    <emb_dir>/<tag>/train/<id>.pt            one per record, foldered by split
    <emb_dir>/<tag>/val/<id>.pt
    <emb_dir>/<tag>/test/<id>.pt

Each ``.pt`` holds ``{embedding: [L, H] float32, shape, **meta}`` where meta
carries the per-structure provenance (embedder, model_name, length, dim,
chains, chain_separator_token).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Repo root = three levels up from this file (workflow/scripts/embed_common.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

SEP = ","

# AbLang2's mask symbol. Standing in for the heavy chain is what makes the
# antibody context leak-free: the model conditions on the light chain and the
# antigen, and predicts the heavy chain.
MASK = "*"

# seq_source -> (heavy column, light column, antigen column)
_SEQ_COLUMNS = {
    "resolved": ("resolved_H_seq", "resolved_L_seq", "resolved_ag_seq"),
    "expected": ("expected_heavy_seq", "expected_light_seq", "expected_ag_seq"),
}


# --- records.csv access ------------------------------------------------------
def load_row(records_csv: str, record_id: str) -> dict:
    """Return the records.csv row for ``record_id`` as a dict of strings."""
    with open(records_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["id"] == record_id:
                return row
    raise KeyError(f"id {record_id!r} not found in {records_csv}")


def _columns(seq_source: str):
    if seq_source not in _SEQ_COLUMNS:
        raise ValueError(
            f"seq_source must be one of {sorted(_SEQ_COLUMNS)}, got {seq_source!r}"
        )
    return _SEQ_COLUMNS[seq_source]


def antigen_sequences(row: dict, seq_source: str) -> dict:
    """Ordered {chain_id: sequence} for the antigen chains of this record."""
    _, _, ag_col = _columns(seq_source)
    chains = [c for c in row["antigen_chains"].split(SEP) if c]
    seqs = row[ag_col].split(SEP)
    if len(chains) != len(seqs):
        raise ValueError(
            f"{row['id']}: {len(chains)} antigen chains but {len(seqs)} sequences "
            f"in column {ag_col}"
        )
    return dict(zip(chains, seqs))


def antibody_sequences(row: dict, seq_source: str) -> tuple:
    """(heavy, light) sequences for this record ('' if a chain is absent).

    The HEAVY sequence returned here is the prediction TARGET. It may be used to
    build training labels; it must never reach an embedder that feeds the model.
    Use ``antibody_context_sequences`` for anything the model conditions on.
    """
    h_col, l_col, _ = _columns(seq_source)
    return row[h_col], row[l_col]


def light_sequence(row: dict, seq_source: str) -> str:
    """The light chain -- the only antibody chain the model is allowed to see."""
    _, l_col, _ = _columns(seq_source)
    light = row[l_col]
    if not light:
        raise ValueError(
            f"{row['id']}: no light chain in column {l_col}. The model conditions "
            "on the antigen and the light chain, so a record without a light "
            "chain cannot be embedded -- set processing.require_paired: true."
        )
    return light


def antibody_context_sequences(row: dict, seq_source: str) -> tuple:
    """(heavy_slot, light) for AbLang2, with the heavy slot MASKED.

    Returns ``('*', light)``: AbLang2 sees a masked heavy chain that absorbs
    antigen/light context, so the resulting embedding encodes what the model is
    given, never what it must predict.
    """
    return MASK, light_sequence(row, seq_source)


# --- output ------------------------------------------------------------------
def save_embedding(path: str, embedding: "torch.Tensor", meta: dict) -> None:
    """Save a per-structure embedding as {embedding:[L,H] float32 cpu, **meta}."""
    import torch

    emb = embedding.detach().to("cpu", torch.float32).contiguous()
    if emb.dim() == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)  # embedders return (1, L, H); store (L, H)
    if emb.dim() != 2:
        raise ValueError(f"expected a 2-D [L, H] embedding, got shape {tuple(emb.shape)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"embedding": emb, "shape": list(emb.shape), **meta}
    torch.save(payload, path)
    print(f"saved {tuple(emb.shape)} -> {path}", flush=True)


def build_meta(
    embedder: str,
    model_name: str,
    matrix: "torch.Tensor",
    chains,
    chain_separator_token: bool,
    **extra,
) -> dict:
    """The per-structure metadata block carried inside every .pt."""
    mat = matrix.squeeze(0) if matrix.dim() == 3 and matrix.shape[0] == 1 else matrix
    meta = {
        "embedder": embedder,
        "model_name": model_name,
        "length": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "chains": list(chains),
        "chain_separator_token": bool(chain_separator_token),
    }
    meta.update(extra)
    return meta


def write_embedder_config(path: str, tag: str, spec: dict, seq_source: str, **extra):
    """Write the one-per-embedder config JSON that sits beside the split dirs."""
    payload = {
        "tag": tag,
        "method": spec.get("method"),
        "class": spec.get("class"),
        "label": spec.get("label", tag),
        "params": {k: v for k, v in spec.items() if k not in ("method", "class", "label")},
        "seq_source": seq_source,
    }
    payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote embedder config -> {path}", flush=True)


# --- ProteinMPNN import without pulling the heavy mango package ----------------
def import_mango_mpnn():
    """Import mango.utils.MPNN_* modules WITHOUT executing mango/__init__.py.

    mango/__init__.py imports the full model stack (esm, ablang2, pyrosetta),
    which we must not require in the minimal ProteinMPNN env. We pre-register
    lightweight stub packages for ``mango`` and ``mango.utils`` pointing at the
    real source dirs, so importing the MPNN submodules resolves their absolute
    ``from mango.utils... import`` statements without running the package inits.
    """
    import importlib
    import types

    if "mango.utils.MPNN_embeddings" in sys.modules:
        return sys.modules["mango.utils.MPNN_embeddings"]

    mango_dir = REPO_ROOT / "mango"
    utils_dir = mango_dir / "utils"

    if "mango" not in sys.modules:
        pkg = types.ModuleType("mango")
        pkg.__path__ = [str(mango_dir)]
        sys.modules["mango"] = pkg
    if "mango.utils" not in sys.modules:
        upkg = types.ModuleType("mango.utils")
        upkg.__path__ = [str(utils_dir)]
        sys.modules["mango.utils"] = upkg

    return importlib.import_module("mango.utils.MPNN_embeddings")
