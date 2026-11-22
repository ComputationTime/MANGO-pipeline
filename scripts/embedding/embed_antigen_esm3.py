"""ESM3 antigen embeddings from sequence or sequence plus resolved backbone.

The sequence baseline encodes chains independently. The structural variant
encodes only the requested antigen chains together in their original frame,
with explicit chain breaks. Antibody chains never enter either representation.
"""

import hashlib
import sys
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SCRIPTS)] + [
    str(_SCRIPTS / d) for d in ("lib", "data", "embedding", "training")
]
import embed_common as ec
import device_common as dc


def antigen_structure(row, seq_source, cif_path=None):
    """Build aligned atom37 coordinates from antigen-only N/CA/C/O atoms."""
    from Bio.PDB import MMCIFParser
    from Bio.Data.PDBData import protein_letters_3to1_extended

    if seq_source != 'resolved':
        raise ValueError('ESM3 sequence_structure requires resolved antigen sequences')
    seqs = ec.antigen_sequences(row, seq_source)
    if not seqs or any(not seq for seq in seqs.values()):
        raise ValueError(f"{row['id']}: empty antigen chain")
    path = cif_path or row.get('pdb_path')
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"{row['id']}: antigen structure does not exist: {path}")
    structure = MMCIFParser(QUIET=True, auth_chains=True).get_structure(row['id'], path)
    model = next(structure.get_models())
    pieces = []
    # ESM3's atom37 convention is N, CA, C, CB, O, ... . Side chains are masked.
    atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 4}
    for chain, sequence in seqs.items():
        if chain not in model:
            raise ValueError(f"{row['id']}: missing ESM3 antigen chain {chain}")
        residues = [residue for residue in model[chain] if not residue.id[0].strip()]
        actual = ''.join(protein_letters_3to1_extended.get(r.resname.strip().upper(), 'X') for r in residues)
        if actual != sequence:
            raise ValueError(f"{row['id']}: ESM3 backbone/sequence mismatch for antigen chain {chain}: "
                             f"{len(sequence)} declared residues, {len(actual)} observed residues")
        coords = torch.full((len(sequence), 37, 3), torch.nan, dtype=torch.float32)
        for position, residue in enumerate(residues):
            for atom, index in atom_indices.items():
                if atom in residue:
                    value = torch.tensor(residue[atom].coord, dtype=torch.float32)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"{row['id']}: nonfinite antigen backbone in chain {chain}")
                    coords[position, index] = value
        if not torch.isfinite(coords[:, :3]).all(dim=-1).all(dim=-1).any():
            raise ValueError(f"{row['id']}: no complete N/CA/C backbone in antigen chain {chain}")
        if pieces:
            pieces.append(torch.full((1, 37, 3), torch.nan, dtype=torch.float32))
        pieces.append(coords)
    return '|'.join(seqs.values()), torch.cat(pieces), list(seqs)


def load_runtime(model_name):
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as exc:
        raise ImportError(
            "ESM3 requires EvolutionaryScale's esm package; use "
            "envs/embed_esm3.yaml"
        ) from exc

    device = dc.get_device(f"ESM3 {model_name} embedding")
    model = ESM3.from_pretrained(model_name, device=device).eval()
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    return model, ESMProtein, logits_config


def esm3(records_csv, record_id, out, seq_source, model_name, tag,
         row=None, runtime=None, representation='sequence', cif_path=None):
    row = row if row is not None else ec.load_row(records_csv, record_id)
    seqs = ec.antigen_sequences(row, seq_source)
    if representation not in {'sequence', 'sequence_structure'}:
        raise ValueError(f'Unsupported ESM3 representation: {representation!r}')
    structural_input = (antigen_structure(row, seq_source, cif_path)
                        if representation == 'sequence_structure' else None)
    if runtime is None:
        model, ESMProtein, logits_config = load_runtime(model_name)
    else:
        model, ESMProtein, logits_config = runtime
    pieces = []
    with torch.no_grad():
        inputs = ([(structural_input[0], structural_input[1])] if structural_input is not None
                  else [(seq, None) for seq in seqs.values()])
        for seq, coordinates in inputs:
            protein = ESMProtein(sequence=seq, coordinates=coordinates)
            encoded = model.encode(protein)
            if coordinates is not None and (encoded.structure is None or encoded.coordinates is None):
                raise RuntimeError(f'{record_id}: ESM3 failed to encode the antigen structure track')
            output = model.logits(encoded, logits_config)
            rep = output.embeddings
            if rep.dim() == 3:
                rep = rep[0]
            if rep.dim() != 2 or rep.shape[0] != len(seq) + 2 or not torch.isfinite(rep).all():
                raise RuntimeError(f'{record_id}: ESM3 returned invalid or misaligned embeddings')
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
            representation=representation,
            conditioning_scope='antigen_only',
            structure_scope='joint_antigen_chains' if structural_input is not None else None,
            structure_atoms=['N', 'CA', 'C', 'O'] if structural_input is not None else None,
            structure_complete_backbone_residues=(int(torch.isfinite(structural_input[1][:, :3]).all(dim=-1).all(dim=-1).sum())
                                                  if structural_input is not None else None),
            antigen_backbone_sha256=(hashlib.sha256(structural_input[1].numpy().tobytes()).hexdigest()
                                     if structural_input is not None else None),
            contributed_module="mango-embedders",
        ),
    )


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        spec = dict(smk.params.spec)
        esm3(smk.input.records, smk.wildcards.instance, smk.output[0],
             smk.params.seq_source, spec.get("model_name", "esm3-sm-open-v1"),
             smk.wildcards.embedder, representation=spec.get('representation', 'sequence'),
             cif_path=smk.input.cif)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True); p.add_argument("--id", required=True)
    p.add_argument("--out", required=True); p.add_argument("--seq-source", default="resolved")
    p.add_argument("--model-name", default="esm3-sm-open-v1")
    p.add_argument("--tag", default="esm3")
    p.add_argument('--representation', choices=['sequence', 'sequence_structure'], default='sequence')
    p.add_argument('--cif')
    a = p.parse_args()
    esm3(a.records, a.id, a.out, a.seq_source, a.model_name, a.tag,
         representation=a.representation, cif_path=a.cif)


if __name__ == "__main__":
    main()
