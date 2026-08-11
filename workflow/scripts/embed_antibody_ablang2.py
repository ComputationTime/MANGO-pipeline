"""AbLang2 antibody CONTEXT embedding, one file per structure.

The task is: given the antigen and the light chain, predict the heavy chain. So
what gets embedded here is the light chain with the heavy slot MASKED -- the
sequence handed to AbLang2 is ``'*|L'``, never ``'H|L'``. This mirrors
mango.MANGORunner's ``ab_seq_context`` convention, where the chain to be
generated is replaced by the mask token; AbLang2 attends to the mask position
freely, so it acts as a slot that absorbs light-chain context.

That masking is the structural guarantee against leakage: the heavy chain is
absent from every artifact the model conditions on, so no downstream stage
(train, evaluate, predict, generate) has to remember to hide it.

We take AbRep's last hidden states. The workflow stages the pinned paired-model
weights once under ``artifacts/weights`` and passes that explicit local path to
``ablang2``; embed workers never download weights. Output shape: [m, 480], 480
= MANGO's hidden size.

The antibody side is deliberately held CONSTANT across the whole study -- every
trained model shares this representation, so any difference between models is
attributable to the antigen representation alone.
"""

import sys
from pathlib import Path

import ablang2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec
import device_common as dc


def ablang2_embed(
    records_csv: str,
    record_id: str,
    out: str,
    seq_source: str,
    model_dir: str,
    row: dict | None = None,
    runtime=None,
) -> None:
    row = row if row is not None else ec.load_row(records_csv, record_id)
    heavy_slot, light = ec.antibody_context_sequences(row, seq_source)

    if runtime is None:
        device = dc.get_device("AbLang2 context embedding")
        model = ablang2.pretrained(
            model_to_use=model_dir, random_init=False, ncpu=1, device=device
        )
    else:
        model, device = runtime

    # AbLang2 pairs the two chains with '|'; the heavy slot is the mask token.
    seqs = [f"{heavy_slot}|{light}"]
    tokens = model.tokenizer(seqs, pad=True, w_extra_tkns=False, device=device)
    with torch.no_grad():
        emb = model.AbRep(tokens).last_hidden_states[0]  # (m, 480)

    ec.save_embedding(
        out,
        emb,
        meta=ec.build_meta(
            embedder="ablang2",
            model_name="ablang2-paired",
            matrix=emb,
            chains=["L"],
            chain_separator_token=True,
            id=record_id,
            seq_source=seq_source,
            context="light_only",
            masked_chains=["H"],
        ),
    )


def load_runtime(model_dir: str):
    """Load AbLang2 once for a multi-record embedding batch."""
    device = dc.get_device("AbLang2 context embedding")
    model = ablang2.pretrained(
        model_to_use=model_dir, random_init=False, ncpu=1, device=device
    )
    return model, device


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        ablang2_embed(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            out=smk.output[0],
            seq_source=smk.params.seq_source,
            model_dir=smk.params.model_dir,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--id", required=True, dest="record_id")
    p.add_argument("--out", required=True)
    p.add_argument("--seq-source", default="resolved")
    p.add_argument("--model-dir", required=True)
    a = p.parse_args()
    ablang2_embed(a.records, a.record_id, a.out, a.seq_source, a.model_dir)


if __name__ == "__main__":
    main()
