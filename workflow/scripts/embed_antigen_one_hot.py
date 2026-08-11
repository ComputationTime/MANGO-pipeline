"""One-hot antigen embedding (H=21), one file per structure.

Mirrors mango.utils.Ag_structure_embeddings.Ag_embeddings.One_hot: antigen
chains are joined with a '|' chain-break token and one-hot encoded over the
21-symbol vocabulary '|ACDEFGHIKLMNPQRSTVWY'. Non-standard residues map to an
all-zero row. Output shape: [L, 21] where L = sum(chain_len) + (n_chains - 1).

This is the naive baseline the study exists to test.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec

VOCAB = "|ACDEFGHIKLMNPQRSTVWY"


def one_hot(
    records_csv: str, record_id: str, out: str, seq_source: str, tag: str,
    row: dict | None = None,
) -> None:
    row = row if row is not None else ec.load_row(records_csv, record_id)
    seqs = ec.antigen_sequences(row, seq_source)
    full_seq = "|".join(seqs.values())

    mat = torch.zeros((len(full_seq), len(VOCAB)), dtype=torch.float32)
    for i, aa in enumerate(full_seq):
        j = VOCAB.find(aa)
        if j >= 0:
            mat[i, j] = 1.0

    ec.save_embedding(
        out,
        mat,
        meta=ec.build_meta(
            embedder=tag,
            model_name="one_hot",
            matrix=mat,
            chains=list(seqs.keys()),
            chain_separator_token=True,
            id=record_id,
            seq_source=seq_source,
        ),
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        one_hot(
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
    p.add_argument("--tag", default="one_hot")
    a = p.parse_args()
    one_hot(a.records, a.record_id, a.out, a.seq_source, a.tag)


if __name__ == "__main__":
    main()
