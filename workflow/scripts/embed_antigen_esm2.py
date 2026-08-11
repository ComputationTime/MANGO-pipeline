"""ESM2 antigen embedding, one file per structure.

Mirrors mango.utils.Ag_structure_embeddings.Ag_embeddings.ESM2: each antigen
chain is embedded separately; per chain we keep the leading <cls> token plus its
residue embeddings and drop the trailing <eos>, concatenate the chains, then
drop the very first <cls>. Net effect: a <cls> embedding sits between chains as
a learned chain-break marker, matching the One_hot '|' separator convention, so
every representation has the same L for the same structure.

Output shape: [L*, H] with L* = sum(chain_len) + (n_chains - 1).
H is 320/480/640/1280/2560/5120 for t6_8M .. t48_15B.

Weights are downloaded automatically by ``fair-esm`` on first use (cached under
~/.cache/torch/hub).
"""

import sys
from pathlib import Path

import esm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec
import device_common as dc

ESM2_SIZES = {"t6_8M", "t12_35M", "t30_150M", "t33_650M", "t36_3B", "t48_15B"}


def _load(size: str):
    if size not in ESM2_SIZES:
        raise ValueError(f"unknown ESM2 size {size!r}; expected one of {sorted(ESM2_SIZES)}")
    model, alphabet = getattr(esm.pretrained, f"esm2_{size}_UR50D")()
    last_layer = int(size.split("_")[0][1:])  # 't6_8M' -> 6
    return model, alphabet, last_layer


def load_runtime(size: str):
    """Load and place ESM2 once for a multi-record embedding batch."""
    device = dc.get_device(f"ESM2 {size} embedding")
    model, alphabet, last_layer = _load(size)
    return model.to(device).eval(), alphabet, last_layer, device


def esm2(
    records_csv: str, record_id: str, out: str, seq_source: str, size: str, tag: str,
    row: dict | None = None, runtime=None,
) -> None:
    row = row if row is not None else ec.load_row(records_csv, record_id)
    seqs = ec.antigen_sequences(row, seq_source)

    if runtime is None:
        model, alphabet, last_layer, device = load_runtime(size)
    else:
        model, alphabet, last_layer, device = runtime
    batch_converter = alphabet.get_batch_converter()

    data = list(seqs.items())          # [(chain_id, seq), ...]
    lens = [len(s) for s in seqs.values()]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        rep = model(tokens.to(device), repr_layers=[last_layer], return_contacts=False)
        reps = rep["representations"][last_layer][:, :-1, :]          # drop trailing <eos>
        per_chain = [emb[: l + 1, :] for emb, l in zip(reps, lens)]   # <cls> + residues
        emb = torch.cat(per_chain, dim=0)[1:, :]                      # drop leading <cls>

    ec.save_embedding(
        out,
        emb,
        meta=ec.build_meta(
            embedder=tag,
            model_name=f"esm2_{size}_UR50D",
            matrix=emb,
            chains=list(seqs.keys()),
            chain_separator_token=True,
            id=record_id,
            seq_source=seq_source,
        ),
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        esm2(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            out=smk.output[0],
            seq_source=smk.params.seq_source,
            size=dict(smk.params.spec)["size"],
            tag=smk.wildcards.embedder,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--id", required=True, dest="record_id")
    p.add_argument("--out", required=True)
    p.add_argument("--seq-source", default="resolved")
    p.add_argument("--size", default="t6_8M")
    p.add_argument("--tag", default="esm2")
    a = p.parse_args()
    esm2(a.records, a.record_id, a.out, a.seq_source, a.size, a.tag)


if __name__ == "__main__":
    main()
