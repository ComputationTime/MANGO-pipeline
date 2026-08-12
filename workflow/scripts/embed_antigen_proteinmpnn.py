"""ProteinMPNN structural antigen embedding, one file per structure.

Uses the ProteinMPNN encoder from mango.utils (imported without pulling the
heavy mango package) to produce per-residue structure embeddings for the
antigen chains only. Coordinates come from the .cif via a Biopython backbone
parser (the bundled ProteinMPNN parser is PDB-only), so the antibody chains and
any HETATM are excluded -- a pure antigen-only representation.

Output shape: [L, H] where L = number of backbone-complete antigen residues and
H is the encoder hidden dim (128 for the vanilla weights).
"""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import embed_common as ec
import mmcif
import device_common as dc

# ProteinMPNN's StructureDatasetPDB drops sequences longer than max_length;
# antigens are already capped upstream, but keep this generous.
MPNN_MAX_LENGTH = 20000
MPNN_CHAIN_ALIASES = tuple(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


def alias_proteinmpnn_chains(pdb_dict: dict, chains: list[str]) -> dict:
    """Return a ProteinMPNN dict whose chain identifiers are one character.

    SAbDab2 author chain identifiers may contain multiple characters (for
    example ``D1``).  ProteinMPNN's contributed featurizer discovers chains by
    taking the final character of each ``seq_chain_*`` key, so feeding author
    identifiers directly can make it look up a nonexistent key.  Alias only
    this private model input; output metadata continues to use author IDs.
    """
    present = [chain for chain in chains if f"seq_chain_{chain}" in pdb_dict]
    if len(present) > len(MPNN_CHAIN_ALIASES):
        raise ValueError(
            f"ProteinMPNN supports at most {len(MPNN_CHAIN_ALIASES)} chains; "
            f"received {len(present)}"
        )

    result = {
        key: value
        for key, value in pdb_dict.items()
        if not key.startswith(("seq_chain_", "coords_chain_"))
    }
    for chain, alias in zip(present, MPNN_CHAIN_ALIASES):
        result[f"seq_chain_{alias}"] = pdb_dict[f"seq_chain_{chain}"]
        source = pdb_dict[f"coords_chain_{chain}"]
        result[f"coords_chain_{alias}"] = {
            key.replace(f"_chain_{chain}", f"_chain_{alias}"): value
            for key, value in source.items()
        }
    result["num_of_chains"] = len(present)
    result["seq"] = "".join(result[f"seq_chain_{a}"] for a in MPNN_CHAIN_ALIASES[:len(present)])
    return result


def load_runtime(weights: str, model: str, noise: int):
    device = dc.get_device(f"ProteinMPNN {model} embedding")
    mp = ec.import_mango_mpnn()
    mp.CHECKPOINTS.setdefault(model, {})[int(noise)] = weights
    encoder = mp.ProteinMPNN_Encoder(
        model=model, noise=int(noise), bb_perturbation=0.0
    )
    # The contributed class records a CUDA ``self.device`` and loads the
    # checkpoint with that map location, but constructs its nn.Module layers
    # on CPU and never moves them.  Featurization correctly follows
    # ``encoder.device``, so explicitly colocate the module and its inputs.
    encoder = encoder.eval().to(device)
    encoder.device = device
    return mp, encoder


def proteinmpnn(
    records_csv: str,
    record_id: str,
    cif_path: str,
    out: str,
    weights: str,
    model: str,
    noise: int,
    tag: str,
    row: dict | None = None,
    runtime=None,
) -> None:
    row = row if row is not None else ec.load_row(records_csv, record_id)
    chains = [c for c in row["antigen_chains"].split(ec.SEP) if c]

    if runtime is None:
        mp, encoder = load_runtime(weights, model, noise)
    else:
        mp, encoder = runtime

    pdb_dict = alias_proteinmpnn_chains(
        mmcif.backbone_dict(cif_path, chains, name=record_id), chains
    )
    dataset = mp.StructureDatasetPDB(
        [pdb_dict], verbose=False, truncate=None, max_length=MPNN_MAX_LENGTH
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{record_id}: structure discarded by StructureDatasetPDB")

    batch = [copy.deepcopy(dataset[0])]
    X, S, mask, residue_idx, chain_encoding_all = mp.tied_featurize_minimal(
        batch, encoder.device, ca_only=encoder.ca_only
    )
    with torch.no_grad():
        h_V = encoder._encode(X, S, mask, residue_idx, chain_encoding_all)  # (1,L,H)

    emb = h_V[0]  # (L, H)
    ec.save_embedding(
        out,
        emb,
        meta=ec.build_meta(
            embedder=tag,
            model_name=f"proteinmpnn_{model}_v_48_{int(noise):03d}",
            matrix=emb,
            chains=chains,
            chain_separator_token=False,
            id=record_id,
            seq_source="structure",
        ),
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        spec = dict(smk.params.spec)
        proteinmpnn(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            cif_path=smk.input.cif,
            out=smk.output[0],
            weights=smk.input.weights,
            model=spec["model"],
            noise=int(spec["noise"]),
            tag=smk.wildcards.embedder,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--id", required=True, dest="record_id")
    p.add_argument("--cif", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="vanilla_models")
    p.add_argument("--noise", type=int, default=2)
    p.add_argument("--tag", default="proteinmpnn")
    a = p.parse_args()
    proteinmpnn(
        a.records, a.record_id, a.cif, a.out, a.weights, a.model, a.noise, a.tag
    )


if __name__ == "__main__":
    main()
