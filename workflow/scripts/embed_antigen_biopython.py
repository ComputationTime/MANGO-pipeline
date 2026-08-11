"""Per-residue biophysical antigen embedding, H=11, one file per structure.

Uses the contributed mango-embedders layout: a chain-level BioPython descriptor
is repeated at every residue, with a zero separator row between chains. This
keeps the sequence axis aligned with residue-wise embedders while remaining
explicit that values within each chain are global properties:

    => L = sum(chain lengths) + n_chains - 1, H = 11

Features, in order:
    0 molecular_weight        4 isoelectric_point      8 sheet_fraction
    1 aromaticity             5 charge_at_pH(7.2)      9 molar_extinction (reduced)
    2 instability_index       6 helix_fraction        10 molar_extinction (cystines)
    3 gravy                   7 turn_fraction

This is one of the two "biophysically informed" representations the grant
contrasts against learned embeddings.
"""

import sys
from pathlib import Path

import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec

HDIM = 11
CHARGE_PH = 7.2  # close to physiological; matches the reference implementation
PRECISION = 4

# ProteinAnalysis rejects anything outside the 20 canonical residues.
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def _clean(seq: str) -> str:
    """Drop non-canonical residues so ProtParam can score the chain."""
    return "".join(c for c in seq.upper() if c in CANONICAL)


def _features(seq: str) -> list:
    """The 11 global biophysical descriptors for one chain."""
    clean = _clean(seq)
    if not clean:
        return [0.0] * HDIM

    x = ProteinAnalysis(clean)
    values = [
        x.molecular_weight(),
        x.aromaticity(),
        x.instability_index(),
        x.gravy(),
        x.isoelectric_point(),
        x.charge_at_pH(CHARGE_PH),
    ]
    values += list(x.secondary_structure_fraction())   # helix, turn, sheet
    values += list(x.molar_extinction_coefficient())   # reduced, cystines
    if len(values) != HDIM:
        raise AssertionError(f"expected {HDIM} features, built {len(values)}")
    return [round(float(v), PRECISION) for v in values]


def biopython(
    records_csv: str, record_id: str, out: str, seq_source: str, tag: str,
    row: dict | None = None,
) -> None:
    row = row if row is not None else ec.load_row(records_csv, record_id)
    seqs = ec.antigen_sequences(row, seq_source)

    rows = []
    for i, seq in enumerate(seqs.values()):
        if i:
            rows.append([0.0] * HDIM)  # chain-break separator row
        descriptor = _features(seq)
        rows.extend([descriptor] * len(seq))

    mat = torch.tensor(rows, dtype=torch.float32)

    ec.save_embedding(
        out,
        mat,
        meta=ec.build_meta(
            embedder=tag,
            model_name="biopython_protparam",
            matrix=mat,
            chains=list(seqs.keys()),
            chain_separator_token=True,
            id=record_id,
            seq_source=seq_source,
            axis="residue",
            descriptor_scope="chain_repeated_per_residue",
            contributed_module="mango-embedders",
            charge_ph=CHARGE_PH,
        ),
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        biopython(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            out=smk.output[0],
            seq_source=smk.params.seq_source,
            tag=smk.wildcards.embedder,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--id", required=True, dest="record_id")
    p.add_argument("--out", required=True)
    p.add_argument("--seq-source", default="resolved")
    p.add_argument("--tag", default="biopython")
    a = p.parse_args()
    biopython(a.records, a.record_id, a.out, a.seq_source, a.tag)


if __name__ == "__main__":
    main()
