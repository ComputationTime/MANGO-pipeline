"""Pearson correlation of per-antibody perplexity with published AlphaSeq affinity."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from finalize_openalphaseq_pairing import digest, write_json


def pearson(x, y, verify=False):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite correlation input')
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    x = x - x.mean(); y = y - y.mean()
    r = float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))
    r = float(np.clip(r, -1, 1))
    if verify:
        np.testing.assert_allclose(r, pearsonr(x, y).statistic, atol=1e-12, rtol=0)
    return r


def paired_nll(heavy_loss, heavy_tokens, light_loss, light_tokens):
    """Combine token losses before exponentiation; never average chain perplexities."""
    return (heavy_loss + light_loss) / (heavy_tokens + light_tokens)


def main(study_path, partial=False):
    study = json.loads(Path(study_path).read_text()); root = Path(study['experiment_root'])
    output = root / ('partial' if partial else 'correlations'); output.mkdir(exist_ok=True)
    measurements = json.loads((root / 'measurements_complete.json').read_text())
    for name in ('measurements', 'cohort'):
        if digest(root / f'{name}.sqlite') != measurements[f'{name}_sha256']:
            raise ValueError(f'Frozen {name} changed')
    with sqlite3.connect(root / 'measurements.sqlite') as db:
        pairs = pd.read_sql_query('SELECT * FROM pairs', db)
        required = pd.read_sql_query('SELECT DISTINCT target_id,ntokens FROM required', db)
    with sqlite3.connect(root / 'cohort.sqlite') as db:
        queries = pd.read_sql_query('SELECT id AS query_id,heavy,light,heavy_novel,light_novel,strict_unseen FROM queries', db)
    pairs = pairs.merge(queries, on='query_id', validate='many_to_one')
    info = pd.read_csv(root / 'antigens.csv').set_index('id')
    pairs['family'] = pairs.antigen_id.map(info.family)
    token_map = required.set_index('target_id').ntokens
    pairs['heavy_tokens'] = pairs.heavy.map(token_map)
    pairs['light_tokens'] = pairs.light.map(token_map).fillna(0)
    counts = {}; hashes = {}; results = []; primary_data = []
    missing = []
    for model_name, spec in study['models'].items():
        directory = root / 'scoring'; path = directory / f'{model_name}.sqlite'
        marker = directory / f'{model_name}.complete.json'
        if not marker.exists():
            missing.append(model_name); continue
        complete = json.loads(marker.read_text())
        manifest_path = directory / f'{model_name}.manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (not complete['complete'] or digest(path) != complete['score_database_sha256'] or
                digest(manifest_path) != complete['manifest_sha256'] or
                manifest['checkpoint_sha256'] != digest(Path(spec['directory']) / 'checkpoints/best.pt') or
                manifest['measurements_sha256'] != measurements['measurements_sha256']):
            raise ValueError(f'{model_name} score provenance changed')
        with sqlite3.connect(path) as db:
            scores = pd.read_sql_query('SELECT target_id,antigen_id,loss_sum FROM scores', db)
        working = pairs.copy()
        for chain in ('heavy', 'light'):
            rename = {'target_id':chain, 'loss_sum':f'{chain}_loss'}
            if spec['family'] == 'iglm':
                values = scores.drop(columns='antigen_id').rename(columns=rename)
                working = working.merge(values, on=chain, how='left', validate='many_to_one')
            else:
                values = scores.rename(columns=rename)
                working = working.merge(values, on=[chain,'antigen_id'], how='left', validate='many_to_one')
        working['heavy_nll'] = working.heavy_loss / working.heavy_tokens
        working['combined_nll'] = paired_nll(working.heavy_loss, working.heavy_tokens,
            working.light_loss.fillna(0), working.light_tokens)
        # Every domain in the primary cohort must be absent from training.
        cohorts = {
            'strict_unseen_heavy': (working.strict_unseen.eq(1) & working.heavy_novel.eq(1), 'heavy_nll'),
            'novel_heavy': (working.heavy_novel.eq(1), 'heavy_nll'),
            'strict_unseen_combined': (working.strict_unseen.eq(1), 'combined_nll'),
        }
        for cohort, (mask, score_key) in cohorts.items():
            eligible = working[mask].copy()
            if not np.isfinite(eligible[score_key]).all():
                raise ValueError(f'{model_name}: missing required scores for {cohort}')
            eligible['nll'] = eligible[score_key]
            eligible['perplexity'] = np.exp(eligible.nll)
            for quality in ('all', 'reliable', 'positive'):
                subset = eligible[eligible[f'{quality}_count'] > 0].copy()
                subset['affinity'] = subset[f'{quality}_sum'] / subset[f'{quality}_count']
                groups = [('pooled','all',subset)]
                groups += [('antigen',str(key),g) for key,g in subset.groupby('antigen_id', sort=True)]
                groups += [('family',str(key),g) for key,g in subset.groupby('family', sort=True)]
                for scope, key, group in groups:
                    results.append(dict(model=model_name, cohort=cohort, quality=quality,
                        scope=scope, group=key, pairs=len(group), antibodies=group.query_id.nunique(),
                        pearson_r_perplexity=pearson(group.perplexity,group.affinity,verify=scope=='pooled'),
                        pearson_r_nll=pearson(group.nll,group.affinity,verify=scope=='pooled')))
                centered = subset[['perplexity','nll','affinity']] - subset.groupby('antigen_id')[
                    ['perplexity','nll','affinity']].transform('mean')
                results.append(dict(model=model_name, cohort=cohort, quality=quality,
                    scope='antigen_centered',group='all',pairs=len(subset),antibodies=subset.query_id.nunique(),
                    pearson_r_perplexity=pearson(centered.perplexity,centered.affinity,verify=True),
                    pearson_r_nll=pearson(centered.nll,centered.affinity,verify=True)))
                if cohort == 'strict_unseen_heavy' and quality == 'reliable':
                    primary_data.append((model_name,subset[['perplexity','affinity']].copy()))
        working.to_csv(output / f'{model_name}_matched_pairs.csv.gz', index=False, compression='gzip')
        counts[model_name] = complete['scores']; hashes[model_name] = digest(marker)
    if missing and not partial:
        raise ValueError(f'The full nine-model comparison is incomplete: {missing}')
    if not results:
        raise ValueError('No complete model scores are available')
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / 'pearson_correlations.csv', index=False)
    primary = metrics[(metrics.cohort=='strict_unseen_heavy') & (metrics.quality=='reliable') &
                      (metrics.scope.isin(['pooled','antigen_centered']))]
    primary.to_csv(output / 'primary_correlations.csv', index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(root / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3,3,figsize=(13,11),sharey=True)
    for axis,(name,data) in zip(axes.flat,primary_data):
        axis.hexbin(data.perplexity,data.affinity,gridsize=60,bins='log',mincnt=1)
        value = primary[(primary.model==name) & (primary.scope=='pooled')].pearson_r_perplexity.iloc[0]
        axis.set_title(f'{name}: r={value:+.4f}')
        axis.set_xlabel('Heavy-chain perplexity'); axis.set_ylabel('AlphaSeq log10(Kd / nM)')
    figure.suptitle('All antibody domains unseen in MANGO training; sufficient assay replicates')
    figure.tight_layout(); figure.savefig(output/'perplexity_affinity.png',dpi=180); plt.close(figure)
    summary = dict(complete=not missing, partial=partial, missing_models=missing, models=list(counts),
        primary='strict_unseen_heavy, reliable measurements',
        affinity_scale='published alphaseq_affinity: log10(estimated Kd in nM); lower is stronger',
        perplexity='exp(mean heavy-variable-domain residue plus EOS NLL)',
        repeats='one vote per normalized antibody/antigen pair; mean qualifying log10 Kd',
        cohort_audit_sha256=digest(root/'cohort_audit.json'), measurements=measurements,
        score_marker_sha256=hashes, implementation_sha256=digest(__file__),
        pearson_crosscheck='Pooled and antigen-centered Pearson r verified against scipy.stats.pearsonr',
        limitations=['Sequence-disjoint does not imply lineage-disjoint.',
            'Affinity was measured for the expressed VHH/scFv, while primary perplexity scores its heavy variable domain.',
            'Pooled correlations can reflect antigen/library offsets; inspect per-antigen and centered results.',
            'Repeated library variants are related; no row-independence p-values or confidence intervals are reported.',
            'Structure-conditioned models use separately predicted antigen-only Chai-1 coordinates.'])
    write_json(output/'summary.json',summary)
    lines=['# AlphaSeq perplexity versus binding affinity','',
           'Primary cohort: every expressed antibody domain is absent from the MANGO training split. '
           'Heavy-chain perplexity includes all normalized variable-domain residues and EOS. '
           'Affinity is the published log10(Kd in nM); lower values indicate stronger binding.','',
           '| Model | Pearson r | Antigen-centered r | Pairs |','|---|---:|---:|---:|']
    for name,group in primary.groupby('model',sort=False):
        rows=group.set_index('scope'); p=rows.loc['pooled']; c=rows.loc['antigen_centered']
        lines.append(f'| {name} | {p.pearson_r_perplexity:+.6f} | {c.pearson_r_perplexity:+.6f} | {int(p.pairs):,} |')
    lines += ['','Per-antigen, per-family, NLL, all-finite, binder-only, novel-heavy and paired-domain '
              'sensitivity results are in pearson_correlations.csv.','']
    lines += ['- '+s for s in summary['limitations']]
    (output/'findings.md').write_text('\n'.join(lines)+'\n')
    print(primary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-path',default='artifacts/experiments/openalphaseq_affinity_full_v1/study.json')
    parser.add_argument('--partial',action='store_true')
    main(**vars(parser.parse_args()))
