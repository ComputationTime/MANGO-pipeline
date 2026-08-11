"""ESM-IF1 structural antigen embeddings, one file per structure.

Adapted from the contributed antigen-embedder module. Each antigen chain is
encoded from its N/CA/C backbone with the fair-esm GVP encoder; chains are
concatenated in dataset order with one zero separator row between them.
Output shape: [sum(chain lengths) + n_chains - 1, 512].
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import embed_common as ec
import mmcif
import device_common as dc


def load_runtime(model_name):
    try:
        import esm
        from esm.inverse_folding.util import get_encoder_output
    except ImportError as exc:
        raise ImportError(
            "ESM-IF requires fair-esm and its inverse-folding dependencies; "
            "use workflow/envs/embed_esmif.yaml"
        ) from exc
    if model_name != "esm_if1_gvp4_t16_142M_UR50":
        raise ValueError(f"unsupported ESM-IF model {model_name!r}")
    device = dc.get_device(f"ESM-IF {model_name} embedding")
    model, alphabet = getattr(esm.pretrained, model_name)()
    return model.eval().to(device), alphabet, get_encoder_output


def esmif(records_csv, record_id, cif_path, out, model_name, tag,
          row=None, runtime=None):
    row = row if row is not None else ec.load_row(records_csv, record_id)
    requested = [c for c in row["antigen_chains"].split(ec.SEP) if c]
    backbone = mmcif.backbone_dict(cif_path, requested, name=record_id)
    chains = [c for c in requested if f"coords_chain_{c}" in backbone]
    if not chains:
        raise RuntimeError(f"{record_id}: no antigen backbones available for ESM-IF")

    if runtime is None:
        model, alphabet, get_encoder_output = load_runtime(model_name)
    else:
        model, alphabet, get_encoder_output = runtime

    pieces = []
    with torch.no_grad():
        for index, chain in enumerate(chains):
            coords = backbone[f"coords_chain_{chain}"]
            xyz = np.stack(
                [np.asarray(coords[f"{atom}_chain_{chain}"], dtype=np.float32)
                 for atom in ("N", "CA", "C")],
                axis=1,
            )
            rep = get_encoder_output(model, alphabet, xyz).detach().float().cpu()
            pieces.append(rep)
            if index + 1 < len(chains):
                pieces.append(torch.zeros((1, rep.shape[-1]), dtype=torch.float32))
    emb = torch.cat(pieces, dim=0)
    ec.save_embedding(
        out,
        emb,
        meta=ec.build_meta(
            embedder=tag,
            model_name=model_name,
            matrix=emb,
            chains=chains,
            chain_separator_token=True,
            id=record_id,
            seq_source="structure",
            contributed_module="mango-embedders",
        ),
    )


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        spec = dict(smk.params.spec)
        esmif(smk.input.records, smk.wildcards.instance, smk.input.cif,
              smk.output[0], spec.get("model_name", "esm_if1_gvp4_t16_142M_UR50"),
              smk.wildcards.embedder)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True); p.add_argument("--id", required=True)
    p.add_argument("--cif", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model-name", default="esm_if1_gvp4_t16_142M_UR50")
    p.add_argument("--tag", default="esmif")
    a = p.parse_args()
    esmif(a.records, a.id, a.cif, a.out, a.model_name, a.tag)


if __name__ == "__main__":
    main()
