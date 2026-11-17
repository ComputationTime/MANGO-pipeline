"""Audit stored pairing scores against independent scalar CPU evaluation."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import csv
import json
import platform
from pathlib import Path
import sqlite3
import sys

import numpy as np
import torch
import transformers

from antigen_diagnostics import mango_token_nll, iglm_token_nll
from finalize_openalphaseq_pairing import digest, write_json
from model_common import load_embedding
from score_openalphaseq_pairing import load_models


def verify(study, partial=False):
    root = Path(study['experiment_root']); output = root / 'partial' if partial else root
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'scoring_manifest.json').read_text())
    audit = json.loads((root / 'cohort_audit.json').read_text())
    if manifest['configuration'] != study or audit['cohort_sha256'] != digest(root / 'cohort.sqlite'):
        raise ValueError('Configuration or held-out cohort changed')
    for path, expected in manifest['scoring_implementation'].items():
        if digest(path) != expected:
            raise ValueError(f'Scoring source no longer matches its frozen manifest: {path}')
    raw_manifest = json.loads((root / 'raw_manifest.json').read_text())
    if raw_manifest['raw_sha256'] != digest(root / 'raw.sqlite'):
        raise ValueError('Raw experimental observations changed')
    if not partial:
        completed = json.loads((root / 'scoring_complete.json').read_text())
        summary = json.loads((root / 'summary.json').read_text())
        if not completed['complete'] or not summary['complete'] or summary['scored_queries'] != audit['unique_queries']:
            raise ValueError('Full scoring/reporting is incomplete')
        if completed['scoring_manifest_sha256'] != digest(root / 'scoring_manifest.json'):
            raise ValueError('Completion marker has different score provenance')
    with sqlite3.connect(root / 'cohort.sqlite') as db:
        target_rows = {r[0]:r for r in db.execute('SELECT * FROM targets WHERE seen_training=0')}
        queries = db.execute('SELECT id,heavy,light,heavy_novel,light_novel,strict_unseen,families FROM queries ORDER BY id').fetchall()
    scoredb = sqlite3.connect(f'file:{(root / "scores.sqlite").resolve()}?mode=ro', uri=True)
    if scoredb.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise ValueError('Score SQLite integrity check failed')
    known = {name:{r[0] for r in scoredb.execute('SELECT target_id FROM scores WHERE model=?', (name,))}
             for name in study['models']}
    common = set.intersection(*known.values())
    selected = set()
    for family in ('RBD','TIGIT','PD1','HER2'):
        candidates = sorted({q[1] for q in queries if q[3] and family in json.loads(q[6]) and q[1] in common})
        if candidates:
            selected.update([candidates[0], candidates[len(candidates)//2]])
    # Light chains test independent selectors and domain composites as well.
    selected.update(sorted(t for t,r in target_rows.items() if r[1]=='light' and t in common)[:3])
    if not selected:
        raise ValueError('No complete novel targets available for verification')
    antigen_ids = manifest['antigen_order']
    antigen_rows = {r['id']:r for r in csv.DictReader((root / 'antigens.csv').open())}
    antigens = {key:load_embedding(str(root / 'openalphaseq/embeddings/antigen/esm2' /
                antigen_rows[key]['split'] / f'{key}.pt')) for key in antigen_ids}
    # This is deliberately an independent CPU mathematical cross-check while
    # the CUDA study remains on the GPU; it is not a training/inference fallback.
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    scorers, provenance = load_models(study, antigens, 'cpu')
    if provenance != manifest['checkpoints']:
        raise ValueError('Checkpoint provenance differs from the score cache')
    checked = {}
    for name, scorer in scorers.items():
        max_full, max_cdr = 0., 0.
        for tid in sorted(selected):
            _, chain, sequence, cdr_json, ntokens, ncdr, _ = target_rows[tid]
            row = scoredb.execute('SELECT loss_sum,cdr3_sum FROM scores WHERE model=? AND target_id=?', (name,tid)).fetchone()
            saved, saved_cdr = [np.frombuffer(blob, dtype=np.float64) for blob in row]
            for j, aid in enumerate(antigen_ids):
                with torch.inference_mode():
                    losses = (iglm_token_nll(scorer.model,chain,sequence) if scorer.family=='iglm' else
                              mango_token_nll(scorer.model,antigens[aid],chain,sequence)).double()
                max_full = max(max_full, abs(saved[j]/ntokens - losses.mean().item()))
                if ncdr:
                    cdr = json.loads(cdr_json)
                    max_cdr = max(max_cdr, abs(saved_cdr[j]/ncdr - losses[cdr].mean().item()))
            if scorer.family == 'iglm' and not np.all(saved == saved[0]):
                raise ValueError('IgLM scores depend on antigen')
        if max_full > 3e-6 or max_cdr > 5e-6:
            raise ValueError(f'{name} cache differs from independent scalar scores: {max_full}, {max_cdr}')
        checked[name] = dict(targets=len(selected), antigens=len(antigen_ids),
                             max_full_nll_error=max_full, max_cdr3_nll_error=max_cdr)
        print(name, json.dumps(checked[name]), flush=True)
    scoredb.close()
    if not partial:
        expected_ids = np.array([q[0] for q in queries])
        for name in study['models']:
            with np.load(root / f'{name}_scores.npz') as data:
                np.testing.assert_array_equal(data['query_ids'], expected_ids)
                np.testing.assert_array_equal(data['antigen_ids'], antigen_ids)
                for key in ('heavy_nll','novel_nll','heavy_cdr3_nll','novel_cdr3_nll'):
                    if data[key].shape != (len(queries),len(antigen_ids)) or not np.isfinite(data[key]).all():
                        raise ValueError('Invalid dense score matrix')
                if name == 'iglm':
                    if not np.all(data['heavy_nll'] == data['heavy_nll'][:,:1]):
                        raise ValueError('Dense IgLM matrix has spurious antigen preferences')
        iglm = next(r for r in summary['rankings'] if r['model']=='iglm')
        if iglm['family_balanced_top1'] != .25 or iglm['measured_within_query_auc'] != .5:
            raise ValueError('IgLM tied-score chance handling failed')
    result = dict(complete=not partial, partial=partial, mathematical_crosscheck_device='cpu',
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                     transformers=transformers.__version__),
        models=checked, source_hashes_verified=True, raw_observations_hash_verified=True,
        cohort_hash_verified=True, canonical_training_overlap=audit['canonical_training_overlap_in_scored_domains'],
        scoring_manifest_sha256=digest(root / 'scoring_manifest.json'),
        cohort_audit_sha256=digest(root / 'cohort_audit.json'), implementation_sha256=digest(__file__))
    if not partial:
        result['output_sha256'] = {p.name:digest(p) for p in sorted(root.glob('*_scores.npz'))}
        result['output_sha256'].update({name:digest(root / name) for name in
            ('summary.json','metrics.csv','per_family_metrics.csv','rankings.csv','binder_metrics.csv','specificity_controls.csv')})
    write_json(output / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/config/openalphaseq_pairing.yaml')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if Path(args.config).suffix == '.json':
        study = json.loads(Path(args.config).read_text())
    else:
        import yaml
        study = yaml.safe_load(Path(args.config).read_text())
    verify(study, args.partial)
