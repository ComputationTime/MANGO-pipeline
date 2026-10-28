"""AlphaFold-Multimer (AF-M) antigen representation. STUB (input prep done).

AF-M receives antigen chains only, matching every other representation. Its
per-residue output must describe only the biological input available to MANGO.

`representation` remains configurable for ablation:
  "single"     per-residue single representation (the decided default; H=384 for AF2)
  "pair"       pairwise representation, pooled to per-residue
  "structure"  predict the structure, then encode it with ProteinMPNN/ESM-IF

Chain assembly below is implemented and testable; what is still blocked is the
fold itself:
  1. AF weights + sequence databases, or precomputed MSAs. MSA generation
     dominates runtime; decide precompute-vs-on-the-fly first.
  2. GPU, and almost certainly a container rather than a conda env.
  3. A separate fetch_af_weights / prep_msa rule, analogous to
     fetch_proteinmpnn_weights.
  4. Decide whether to orchestrate AF inside this rule or run it externally and
     have this rule merely INGEST outputs. Ingest is far easier to make
     restartable and is recommended.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec

def build_fold_input(row: dict, seq_source: str) -> dict:
    """Ordered antigen-only ``{chain_label: sequence}`` handed to AF-M."""
    return dict(ec.antigen_sequences(row, seq_source))


def afm(
    records_csv: str,
    record_id: str,
    out: str,
    seq_source: str,
    representation: str,
    tag: str,
) -> None:
    """AF-Multimer representation for antigen chains -> [L, H]."""
    row = ec.load_row(records_csv, record_id)
    chains = build_fold_input(row, seq_source)
    total_len = sum(len(s) for s in chains.values()) + len(chains) - 1

    raise NotImplementedError(
        f"AF-M antigen embedder is not implemented yet ({tag}).\n"
        f"Fold input is ready: {len(chains)} chain(s) "
        f"{sorted(chains)} -> expected L={total_len}, "
        f"representation={representation!r}.\n"
        "Blocked on: MSA strategy, weights, and GPU/container infrastructure.\n"
        "Remove 'afm' from active_embedders in config/config.yaml to run the "
        "rest of the pipeline."
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        spec = dict(smk.params.spec)
        afm(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            out=smk.output[0],
            seq_source=smk.params.seq_source,
            representation=spec.get("representation", "single"),
            tag=smk.wildcards.embedder,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--id", required=True, dest="record_id")
    p.add_argument("--out", required=True)
    p.add_argument("--seq-source", default="resolved")
    p.add_argument("--representation", default="single")
    p.add_argument("--tag", default="afm")
    a = p.parse_args()
    afm(
        a.records, a.record_id, a.out, a.seq_source, a.representation,
        a.tag,
    )


if __name__ == "__main__":
    main()
