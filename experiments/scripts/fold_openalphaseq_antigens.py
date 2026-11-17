"""Predict each exact AlphaSeq antigen alone, without antibody or affinity inputs."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from finalize_openalphaseq_pairing import digest, write_json


def validate_structure(path, sequence):
    from Bio.PDB import MMCIFParser
    from Bio.Data.PDBData import protein_letters_3to1_extended
    model = next(MMCIFParser(QUIET=True, auth_chains=True).get_structure('antigen', path).get_models())
    chains = list(model.get_chains())
    if len(chains) != 1 or chains[0].id != 'A':
        raise ValueError('The antigen-only prediction must contain exactly chain A')
    residues = [r for r in chains[0] if r.id[0] == ' ']
    actual = ''.join(protein_letters_3to1_extended.get(r.resname, 'X') for r in residues)
    if actual != sequence or any(not all(a in r for a in ('N', 'CA', 'C', 'O')) for r in residues):
        raise ValueError('Predicted structure does not cover the exact full antigen sequence/backbone')


def main(study_path, antigen_id=None):
    study = json.loads(Path(study_path).read_text())
    root = Path(study['experiment_root']); spec = study['structure']
    rows = list(csv.DictReader((root / 'antigen_records.csv').open()))
    provenance = dict(method=spec, records_sha256=digest(root / 'antigen_records.csv'),
                      implementation_sha256=digest(__file__), antibody_inputs=False, affinity_inputs=False)
    manifest = root / 'structures_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError('Antigen structure inputs changed; preserve the old inference cache')
    write_json(manifest, provenance)
    if antigen_id:
        row = next(r for r in rows if r['id'] == antigen_id)
        destination = Path(row['pdb_path']); destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            validate_structure(destination, row['resolved_ag_seq'])
            fasta = destination.parent / 'antigen.fasta'
            if not fasta.exists():
                raise ValueError('A cached antigen structure has no input sequence provenance')
            write_json(destination.parent / 'complete.json', dict(complete=True, antigen_id=antigen_id,
                sequence_sha256=digest(fasta), structure_sha256=digest(destination),
                structure_manifest_sha256=digest(manifest)))
            return
        os.environ['CHAI_DOWNLOADS_DIR'] = str(Path(spec['cache_dir']).resolve())
        import torch
        from Bio.PDB import MMCIFIO, MMCIFParser
        from chai_lab.chai1 import run_inference
        torch.set_num_threads(2)
        fasta = destination.parent / 'antigen.fasta'
        fasta.write_text(f'>protein|name=antigen\n{row["resolved_ag_seq"]}\n')
        # Chai requires an empty output directory. Preserve failed attempts so
        # an interrupted target can resume without deleting useful diagnostics.
        prediction = destination.parent / 'prediction'
        attempt = 0
        while prediction.exists() and any(prediction.iterdir()):
            attempt += 1
            prediction = destination.parent / f'prediction_retry_{attempt}'
        candidates = run_inference(fasta_file=fasta, output_dir=prediction,
            num_trunk_recycles=spec['recycles'], num_diffn_samples=spec['diffusion_samples'],
            seed=study['seed'], device='cuda:0', use_esm_embeddings=spec['use_esm_embeddings'],
            use_msa_server=spec['use_msa_server'])
        structure = MMCIFParser(QUIET=True).get_structure('antigen', candidates.cif_paths[0])
        chains = list(next(structure.get_models()).get_chains())
        if len(chains) != 1:
            raise ValueError('Unexpected additional chains in antigen prediction')
        chains[0].id = 'A'
        temporary = destination.with_suffix('.tmp.cif')
        writer = MMCIFIO(); writer.set_structure(structure); writer.save(str(temporary))
        validate_structure(temporary, row['resolved_ag_seq']); temporary.replace(destination)
        write_json(destination.parent / 'complete.json', dict(complete=True, antigen_id=antigen_id,
            sequence_sha256=digest(fasta), structure_sha256=digest(destination),
            structure_manifest_sha256=digest(manifest)))
        return
    failed = []
    for index, row in enumerate(rows, 1):
        marker = Path(row['pdb_path']).parent / 'complete.json'
        if marker.exists():
            old = json.loads(marker.read_text())
            if old['structure_manifest_sha256'] != digest(manifest) or old['structure_sha256'] != digest(row['pdb_path']):
                raise ValueError('Cached antigen structure changed')
            validate_structure(row['pdb_path'], row['resolved_ag_seq'])
            continue
        log_path = Path(row['pdb_path']).parent / 'fold.log'; log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as log:
            code = subprocess.run([sys.executable, __file__, '--study-path', study_path,
                                   '--antigen-id', row['id']], stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            failed.append(row['id'])
        write_json(root / 'structures_progress.json', dict(complete=False, visited=index,
                   total=len(rows), failed=failed, last_log=str(log_path)))
        print(f'Antigen structures {index}/{len(rows)}, failures={len(failed)}', flush=True)
        if len(failed) >= 3:
            raise RuntimeError(f'Three antigen prediction failures; inspect {log_path}')
    if failed:
        raise RuntimeError(f'Incomplete antigen predictions: {failed}')
    write_json(root / 'structures_complete.json', dict(complete=True, antigens=len(rows),
               manifest_sha256=digest(manifest), structures={r['id']:digest(r['pdb_path']) for r in rows}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-path', default='artifacts/experiments/openalphaseq_affinity_full_v1/study.json')
    parser.add_argument('--antigen-id')
    main(**vars(parser.parse_args()))
