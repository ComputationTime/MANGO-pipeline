"""Summarize available transfer results, including partial independent runs."""
import argparse
import json
from pathlib import Path
import csv
from datetime import datetime, timezone


def report(root):
    root = Path(root)
    tags = ['esm2', 'control_constant', 'control_random', 'control_constant_l1',
            'control_random_l1', 'control_pooled_esm2', 'control_shuffled', 'control_cross_attention_esm2']
    manifest=root/'experiment_manifest.json'
    configuration=json.loads(manifest.read_text())['configuration'] if manifest.is_file() else {}
    if configuration: tags=configuration['controls']
    audit=root/'cohort'/'split_audit.json'
    source=configuration.get('pretraining_dataset', json.loads(audit.read_text()).get('pretraining_dataset', 'abrank') if audit.is_file() else 'abrank')
    runs, rows = [], []
    for phase in [source, 'sabdab2']:
        for tag in tags:
            directory = root/phase/'runs'/tag
            path = directory/'eval.json'
            entry = dict(phase=phase, embedder=tag, complete=path.is_file(), evaluation=str(path))
            runtime_path=directory/'training_runtime.json'
            runtime=json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
            entry['training_complete']=runtime.get('status')=='complete'
            entry['complete']=entry['complete'] and entry['training_complete']
            metrics=directory/'metrics.jsonl'
            if metrics.is_file():
                history=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
                if history:
                    best=min(history,key=lambda value:value['val_loss'])
                    entry.update(completed_epochs=len(history),best_validation_nll=best['val_loss'],
                                 best_validation_epoch=best['epoch'])
            if path.is_file():
                data = json.loads(path.read_text())
                for split, value in data['splits'].items():
                    rows.append(dict(phase=phase, embedder=tag, split=split, nll=value['nll'],
                                     perplexity=value['perplexity'], n_examples=value['n_examples'],
                                     n_tokens=value['n_tokens']))
            else:
                stage = ('embedding' if not (root/phase/'embeddings'/'antigen'/('esm2' if tag=='control_cross_attention_esm2' else tag)/'complete.json').is_file()
                         else 'training' if not entry['training_complete'] else 'evaluation')
                log_stage = {'embedding':'embed','training':'train','evaluation':'eval'}[stage]
                log_tag='esm2' if stage=='embedding' and tag=='control_cross_attention_esm2' else tag
                entry.update(first_missing_stage=stage, log=str(root/'logs'/f'{phase}_{log_stage}_{log_tag}.log'))
            runs.append(entry)
    figures = [root/'plots'/f'{name}.{ext}' for name in [f'{source}_controls','sabdab2_finetuned_controls'] for ext in ['png','csv']]
    result = dict(updated_at=datetime.now(timezone.utc).isoformat(), runs=runs,
                  figures={str(p):p.is_file() for p in figures},
                  complete=all(r['complete'] for r in runs) and all(p.is_file() for p in figures))
    temporary=root/'results_report.json.tmp'
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(root/'results_report.json')
    with (root/'results_report.csv.tmp').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=['phase','embedder','split','nll','perplexity','n_examples','n_tokens'])
        writer.writeheader();writer.writerows(rows)
    (root/'results_report.csv.tmp').replace(root/'results_report.csv')
    print(f"Transfer experiment: {sum(r['training_complete'] for r in runs)}/{len(runs)} trained, {sum(r['complete'] for r in runs)}/{len(runs)} evaluations; complete={result['complete']}")
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    result=report(args.root)
    if args.require_complete and not result['complete']:raise SystemExit(1)
