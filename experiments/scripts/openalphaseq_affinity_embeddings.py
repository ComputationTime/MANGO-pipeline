"""Use the production antigen embedders unchanged for the external affinity set."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import json
from pathlib import Path

from embed_batch import batch_embed
from finalize_openalphaseq_pairing import digest, write_json


def main(study_path, model_name):
    study = json.loads(Path(study_path).read_text())
    root = Path(study['experiment_root'])
    spec = study['models'][model_name]['spec']
    output = root / 'embeddings' / model_name
    weights = (f'artifacts/weights/proteinmpnn/{spec["model"]}/v_48_{int(spec["noise"]):03d}.pt'
               if spec['method'] == 'proteinmpnn' else None)
    batch_embed(str(root / 'antigen_records.csv'), str(output), str(output / '.batch_complete.json'),
                kind='antigen', tag=model_name, method=spec['method'], spec=spec,
                seq_source='resolved', splits=['test'], implementation=__file__, weights=weights)
    marker = output / '.batch_complete.json'
    details = json.loads(marker.read_text())
    details.update(records_sha256=digest(root / 'antigen_records.csv'),
                   implementation_sha256=digest(__file__),
                   files_sha256={p.stem:digest(p) for p in sorted((output / 'test').glob('*.pt'))})
    if model_name in {'pyrosetta_pre','esm3_structure','esmif','proteinmpnn'}:
        structures = json.loads((root / 'structures_complete.json').read_text())
        for aid, expected in structures['structures'].items():
            if digest(root / 'structures' / aid / 'antigen.cif') != expected:
                raise ValueError('An antigen structure changed after prediction')
        details['structures_marker_sha256'] = digest(root / 'structures_complete.json')
    write_json(marker, details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-path', default='artifacts/experiments/openalphaseq_affinity_full_v1/study.json')
    parser.add_argument('--model-name', required=True)
    main(**vars(parser.parse_args()))
