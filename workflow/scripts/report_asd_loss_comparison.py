"""Write partial objective/route comparisons while the serialized queue runs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def tail_iteration(path):
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        lines = handle.read().decode().splitlines()
    for line in reversed(lines):
        parts = line.split(',')
        if len(parts) > 3 and parts[0].isdigit():
            return int(parts[0])
    return None


def report(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'experiment_manifest.json'
    study = json.loads(manifest.read_text())['configuration'] if manifest.is_file() else {}
    losses = study.get('objectives', {'cross_entropy': {}, 'focal': {}})
    architectures = study.get('architectures', {'mango': '', 'cross_attention': ''})
    runs, rows = [], []
    for phase in ('asd', 'sabdab2_scratch', 'sabdab2_finetuned'):
        for loss in losses:
            for architecture in architectures:
                directory = root / phase / loss / architecture
                evaluation = directory / 'eval.json'
                runtime = directory / 'training_runtime.json'
                trained = runtime.is_file() and json.loads(runtime.read_text()).get('status') == 'complete'
                entry = dict(phase=phase, objective=loss, architecture=architecture,
                             training_complete=trained, complete=trained and evaluation.is_file(),
                             evaluation=str(evaluation), iteration=tail_iteration(directory / 'training_curve.csv'),
                             log=str(root / 'logs' / f'{phase}_{loss}_{architecture}_train.log'))
                if not entry['complete']:
                    dataset = 'asd' if phase == 'asd' else 'sabdab2'
                    embeddings = Path(study.get('embedding_root', root)) / dataset / 'embeddings/antigen/esm2/complete.json'
                    if not embeddings.is_file():
                        entry.update(first_missing_stage='embedding',
                                     log=str(root / 'logs' / f'{dataset}_embed_esm2.log'))
                    elif phase == 'sabdab2_finetuned' and not (root / 'asd' / loss / architecture / 'checkpoints/best.pt').is_file():
                        entry.update(first_missing_stage='pretraining',
                                     log=str(root / 'logs' / f'asd_{loss}_{architecture}_train.log'))
                    elif not trained:
                        entry['first_missing_stage'] = 'training'
                    else:
                        entry.update(first_missing_stage='evaluation',
                                     log=str(root / 'logs' / f'{phase}_{loss}_{architecture}_eval.log'))
                history_path = directory / 'metrics.jsonl'
                if history_path.is_file():
                    history = [json.loads(s) for s in history_path.read_text().splitlines() if s.strip()]
                    if history:
                        best = min(history, key=lambda r: r['val_loss'])
                        entry.update(completed_epochs=len(history), best_validation_nll=best['val_loss'])
                if evaluation.is_file():
                    payload = json.loads(evaluation.read_text())
                    for split, value in payload['splits'].items():
                        rows.append(dict(phase=phase, objective=loss, architecture=architecture,
                            split=split, nll=value['nll'], perplexity=value['perplexity'],
                            n_examples=value['n_examples'], n_tokens=value['n_tokens']))
                runs.append(entry)
    figures = [root / 'plots' / f'sabdab2_loss_comparison.{ext}' for ext in ('png', 'csv')]
    result = dict(updated_at=datetime.now(timezone.utc).isoformat(), runs=runs,
                  completed_final_models=sum(r['complete'] for r in runs if r['phase'] != 'asd'),
                  expected_final_models=8, complete=all(r['complete'] for r in runs) and all(p.is_file() for p in figures))
    temporary = root / 'results_report.json.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(root / 'results_report.json')
    with (root / 'results_report.csv.tmp').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['phase', 'objective', 'architecture', 'split', 'nll', 'perplexity', 'n_examples', 'n_tokens'])
        writer.writeheader()
        writer.writerows(rows)
    (root / 'results_report.csv.tmp').replace(root / 'results_report.csv')
    print(f"ESM2 objective comparison: {sum(r['training_complete'] for r in runs)}/12 stages trained; "
          f"{result['completed_final_models']}/8 final models evaluated; complete={result['complete']}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = report(args.root)
    if args.require_complete and not result['complete']:
        raise SystemExit(1)
