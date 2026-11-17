"""Report local HER2/PD1/TIGIT correlations from frozen nine-model likelihoods."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finalize_openalphaseq_pairing import digest, write_json
from report_openalphaseq_affinity import main as affinity_report


def main(study_path, partial=False):
    study = json.loads(Path(study_path).read_text()); root = Path(study['experiment_root'])
    affinity_report(study_path, partial=partial)
    output = root / ('partial' if partial else 'correlations')
    metrics = pd.read_csv(output / 'pearson_correlations.csv')
    info = pd.read_csv(root / 'antigens.csv').set_index('id')
    primary = metrics[(metrics.cohort == 'strict_unseen_heavy') & (metrics.quality == 'reliable') &
                      metrics.scope.isin(['pooled', 'antigen', 'antigen_centered'])].copy()
    primary['target'] = primary['group'].map(info.family)
    primary.loc[primary.scope == 'pooled', 'target'] = 'pooled'
    primary.loc[primary.scope == 'antigen_centered', 'target'] = 'antigen_centered'
    primary.to_csv(output / 'primary_correlations.csv', index=False)
    table = primary.pivot(index='model', columns='target', values='pearson_r_perplexity')
    table = table.reindex(columns=['HER2', 'PD1', 'TIGIT', 'pooled', 'antigen_centered'])
    table.to_csv(output / 'pearson_by_model_target.csv')
    novel = metrics[(metrics.cohort == 'novel_heavy') & (metrics.quality == 'reliable') &
                    metrics.scope.isin(['pooled', 'antigen', 'antigen_centered'])].copy()
    novel['target'] = novel['group'].map(info.family)
    novel.loc[novel.scope == 'pooled', 'target'] = 'pooled'
    novel.loc[novel.scope == 'antigen_centered', 'target'] = 'antigen_centered'
    novel_table = novel.pivot(index='model', columns='target', values='pearson_r_perplexity')
    novel_table = novel_table.reindex(columns=table.columns)
    novel_table.to_csv(output / 'novel_heavy_pearson_by_model_target.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = {(cohort,family):plt.subplots(3, 3, figsize=(13, 11))
               for cohort in ('strict_unseen_heavy', 'novel_heavy') for family in info.family.unique()}
    counts = {}; novel_counts = {}; provenance = {}
    for index, name in enumerate(table.index):
        path = output / f'{name}_matched_pairs.csv.gz'
        pairs = pd.read_csv(path)
        pairs['target'] = pairs.antigen_id.map(info.family)
        pairs['heavy_perplexity'] = np.exp(pairs.heavy_nll)
        # A seen light chain is intentionally unscored; do not label its absent
        # likelihood as zero loss in the exported paired-domain sensitivity.
        missing_light = pairs.light.notna() & pairs.light_loss.isna()
        pairs.loc[missing_light, 'combined_nll'] = np.nan
        pairs['combined_perplexity'] = np.exp(pairs.combined_nll)
        pairs['reliable_affinity'] = pairs.reliable_sum / pairs.reliable_count.replace(0, np.nan)
        pairs['primary_eligible'] = (pairs.strict_unseen.eq(1) & pairs.heavy_novel.eq(1) & pairs.reliable_count.gt(0))
        pairs['novel_heavy_eligible'] = pairs.heavy_novel.eq(1) & pairs.reliable_count.gt(0)
        pairs.to_csv(path, index=False, compression='gzip')
        eligible = pairs[pairs.primary_eligible]
        counts[name] = eligible.groupby('target').size().reindex(info.family.unique(), fill_value=0).to_dict()
        novel_eligible = pairs[pairs.novel_heavy_eligible]
        novel_counts[name] = novel_eligible.groupby('target').size().reindex(info.family.unique(), fill_value=0).to_dict()
        spec = study['models'][name]
        checkpoint = Path(spec['directory']) / 'checkpoints/best.pt'
        if digest(checkpoint) != spec['checkpoint_sha256']:
            raise ValueError(f'{name}: frozen checkpoint changed')
        provenance[name] = dict(directory=spec['directory'], checkpoint_sha256=digest(checkpoint),
                                checkpoint_epoch=spec['checkpoint_epoch'])
        for (cohort,family), (figure, axes) in figures.items():
            cohort_data = eligible if cohort == 'strict_unseen_heavy' else novel_eligible
            axis = axes.flat[index]; data = cohort_data[cohort_data.target == family]
            if len(data):
                axis.hexbin(data.heavy_perplexity, data.reliable_affinity, gridsize=50, bins='log', mincnt=1)
            value = (table if cohort == 'strict_unseen_heavy' else novel_table).loc[name, family]
            axis.set_title(f'{name}: r={value:+.4f}; n={len(data):,}')
            axis.set_xlabel('Heavy-chain perplexity'); axis.set_ylabel('AlphaSeq log10(Kd / nM)')
    for (cohort,family), (figure, axes) in figures.items():
        for axis in list(axes.flat)[len(table):]:
            axis.set_visible(False)
        label = 'all expressed domains unseen' if cohort == 'strict_unseen_heavy' else 'heavy domain unseen; light may be training-seen'
        figure.suptitle(f'{family}: {label}; sufficient assay replicates')
        figure.tight_layout(); figure.savefig(output / f'{family}_{cohort}_perplexity_affinity.png', dpi=180); plt.close(figure)
    summary = json.loads((output / 'summary.json').read_text())
    summary.update(source_manifest_sha256=digest(root / 'raw_manifest.json'),
                   source_files=study['local_binding_inputs']['source_sha256'],
                   primary_pairs_by_target=counts, checkpoints=provenance,
                   novel_heavy_pairs_by_target=novel_counts,
                   empty_primary_targets=[f for f in info.family.unique() if not any(c.get(f,0) for c in counts.values())],
                   report_implementation_sha256=digest(__file__))
    write_json(output / 'local_summary.json', summary)
    lines = ['# MANGO binding-affinity correlations', '',
        'Pearson r between heavy-variable-domain perplexity and the supplied AlphaSeq affinity '
        '(log10 estimated Kd in nM; lower means stronger binding). All expressed domains are absent '
        'from the training split. Only finite measurements with sufficient replicates contribute; '
        'repeats are averaged per normalized antibody–antigen pair.', '',
        '| Model | HER2 | PD1 | TIGIT | Pooled | Antigen-centered |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in table.iterrows():
        values = ['undefined' if pd.isna(v) else f'{v:+.6f}' for v in row]
        lines.append('| ' + ' | '.join([name] + values) + ' |')
    lines += ['', 'HER2 and PD1 have no fully unseen constructs: their expressed light chains occur '
              'in training. Their strict correlations are undefined. The following sensitivity '
              'analysis requires an unseen heavy domain and permits a training-seen light domain.', '',
              '| Model | HER2 | PD1 | TIGIT | Pooled | Antigen-centered |',
              '|---|---:|---:|---:|---:|---:|']
    for name, row in novel_table.iterrows():
        values = ['undefined' if pd.isna(v) else f'{v:+.6f}' for v in row]
        lines.append('| ' + ' | '.join([name] + values) + ' |')
    lines += ['', 'Pair counts and checkpoint provenance are in `local_summary.json`. '
              'Per-model matched scores and eligibility flags are in `*_matched_pairs.csv.gz`. '
              'All-finite, paired-domain, above-background and NLL sensitivity results are in '
              '`pearson_correlations.csv`.', '']
    lines += ['- ' + item for item in summary['limitations']]
    if summary['missing_models']:
        lines += ['', 'Incomplete models: ' + ', '.join(summary['missing_models'])]
    (output / 'findings.md').write_text('\n'.join(lines) + '\n')
    print(table.to_string(), flush=True)
    print('Novel-heavy sensitivity (training-seen light chains allowed):', flush=True)
    print(novel_table.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-path', required=True)
    parser.add_argument('--partial', action='store_true')
    main(**vars(parser.parse_args()))
