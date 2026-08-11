"""AlphaFold-Multimer (AF-M) antigen representation.  STUB (input prep done).

DECIDED: AF-M is folded on exactly the information the model is allowed to
condition on -- the ANTIGEN chains plus the LIGHT chain -- and we take its
per-residue `single` representation from the Evoformer stack. The heavy chain,
which is the prediction target, is never given to AF-M, so this representation
cannot leak the answer.

Why fold the light chain at all: AF-M's value over a sequence model is that it
models the INTERFACE. Folding the antigen alone would throw that away and make
AF-M a strictly worse ESM2. Folding antigen + light chain gives a representation
of the partially-assembled complex, which is precisely the context a heavy chain
has to complete -- and it stays inside the conditioning set.

Consequence for shape: this is the one embedder whose L is not the antigen
length. L spans the antigen chains AND the light chain, with the usual one
separator row between chains, so::

    L = sum(len(ag_chain) for each antigen chain) + len(light) + n_chains - 1

Cross-attention imposes no constraint on L (the antigen side is keys/values, of
any length), so nothing downstream breaks -- but do not compare this L against
the other seven embedders' as though it measured the same thing.

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

# The light chain is folded alongside the antigen; 'L' is its chain label in the
# AF-M input. Every other chain label comes from the record's antigen chains.
LIGHT_CHAIN_LABEL = "L"


def build_fold_input(row: dict, seq_source: str, include_light_chain: bool) -> dict:
    """Ordered {chain_label: sequence} handed to AF-M.

    Antigen chains first, in records order, then the light chain. Splitting this
    out from the fold keeps the conditioning set inspectable (and testable)
    without any AF infrastructure: whatever is in this dict is exactly what the
    representation is allowed to know.
    """
    chains = dict(ec.antigen_sequences(row, seq_source))
    if include_light_chain:
        if LIGHT_CHAIN_LABEL in chains:
            raise ValueError(
                f"{row['id']}: antigen chain id {LIGHT_CHAIN_LABEL!r} collides with "
                "the light-chain label used for AF-M input; relabel the antigen "
                "chains for this record before folding."
            )
        chains[LIGHT_CHAIN_LABEL] = ec.light_sequence(row, seq_source)
    return chains


def afm(
    records_csv: str,
    record_id: str,
    out: str,
    seq_source: str,
    representation: str,
    include_light_chain: bool,
    tag: str,
) -> None:
    """AF-Multimer representation for antigen (+ light) chains -> [L, H]."""
    row = ec.load_row(records_csv, record_id)
    chains = build_fold_input(row, seq_source, include_light_chain)
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
            include_light_chain=bool(spec.get("include_light_chain", True)),
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
    p.add_argument(
        "--no-light-chain",
        action="store_true",
        help="fold the antigen alone (ablation; drops the interface signal)",
    )
    p.add_argument("--tag", default="afm")
    a = p.parse_args()
    afm(
        a.records, a.record_id, a.out, a.seq_source, a.representation,
        not a.no_light_chain, a.tag,
    )


if __name__ == "__main__":
    main()
