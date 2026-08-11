"""ESM3 sequence embeddings for antigen chains, one file per structure.

Adapted from the contributed antigen-embedder module. This implementation is
explicitly sequence-only: each chain is encoded independently, its EOS is
dropped, and retained BOS rows become learned separators between chains.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec
import device_common as dc


def load_runtime(model_name):
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as exc:
        raise ImportError(
            "ESM3 requires EvolutionaryScale's esm package; use "
            "workflow/envs/embed_esm3.yaml"
        ) from exc

    device = dc.get_device(f"ESM3 {model_name} embedding")
    model = ESM3.from_pretrained(model_name).to(device).eval()
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    return model, ESMProtein, logits_config


def esm3(records_csv, record_id, out, seq_source, model_name, tag,
         row=None, runtime=None):
    row = row if row is not None else ec.load_row(records_csv, record_id)
    seqs = ec.antigen_sequences(row, seq_source)
    if runtime is None:
        model, ESMProtein, logits_config = load_runtime(model_name)
    else:
        model, ESMProtein, logits_config = runtime
    pieces = []
    with torch.no_grad():
        for seq in seqs.values():
            encoded = model.encode(ESMProtein(sequence=seq))
            output = model.logits(encoded, logits_config)
            rep = output.embeddings
            if rep.dim() == 3:
                rep = rep[0]
            pieces.append(rep[: len(seq) + 1].detach().float().cpu())
    emb = torch.cat(pieces, dim=0)[1:, :]
    ec.save_embedding(
        out,
        emb,
        meta=ec.build_meta(
            embedder=tag,
            model_name=model_name,
            matrix=emb,
            chains=list(seqs),
            chain_separator_token=True,
            id=record_id,
            seq_source=seq_source,
            representation="sequence",
            contributed_module="mango-embedders",
        ),
    )


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        spec = dict(smk.params.spec)
        esm3(smk.input.records, smk.wildcards.instance, smk.output[0],
             smk.params.seq_source, spec.get("model_name", "esm3-sm-open-v1"),
             smk.wildcards.embedder)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True); p.add_argument("--id", required=True)
    p.add_argument("--out", required=True); p.add_argument("--seq-source", default="resolved")
    p.add_argument("--model-name", default="esm3-sm-open-v1")
    p.add_argument("--tag", default="esm3")
    a = p.parse_args()
    esm3(a.records, a.id, a.out, a.seq_source, a.model_name, a.tag)


if __name__ == "__main__":
    main()
