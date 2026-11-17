"""Inventory likelihood-training outputs independently of design-analysis status."""
import argparse
import csv
import json
from pathlib import Path

import yaml

from finalize_openalphaseq_pairing import digest, write_json
from prepare_openalphaseq_affinity import model_directories


def main(config='config/openalphaseq_affinity.yaml'):
    study = yaml.safe_load(Path(config).read_text())
    rows = list(csv.DictReader(Path(study['training_records']).open()))
    counts = {s:2*sum(r['split']==s for r in rows) for s in ('train','val','test')}
    inventory = {}; cohorts = set()
    for name,spec in model_directories(study).items():
        root = Path(spec['directory']); evaluation = root/'eval.json'
        missing = [str(root/f) for f in ['checkpoints/best.pt','model_config.json','eval.json',
                   'training_curve.csv','training_curve.png'] if not (root/f).is_file()]
        info = dict(directory=str(root), complete=not missing, missing=missing)
        if evaluation.exists():
            data = json.loads(evaluation.read_text()); cohorts.add(data['cohort_sha256'])
            info.update(checkpoint_epoch=data['checkpoint_epoch'], splits=data['splits'],
                        cohort_sha256=data['cohort_sha256'],evaluation_sha256=digest(evaluation))
            for split,expected in counts.items():
                actual=data['splits'].get(split,{})
                if actual.get('n_examples')!=expected or actual.get('n_skipped_missing_embeddings',0):
                    info.update(complete=False,error=f'{split} evaluation coverage differs from the full cohort')
        info['logs'] = dict(training=f'artifacts/logs/train_{root.name}.log',
                            evaluation=f'artifacts/logs/evaluate_{root.name}.log')
        inventory[name]=info
    complete=sum(i['complete'] for i in inventory.values())
    report=dict(complete=complete==9 and len(cohorts)==1, models_complete=complete, models_requested=9,
                training_records_sha256=digest(study['training_records']), runs=inventory)
    write_json('artifacts/gpu/train_all_report.json',report)
    print(f'Likelihood training/evaluation: {complete}/9 models complete',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/openalphaseq_affinity.yaml')
    main(**vars(parser.parse_args()))
