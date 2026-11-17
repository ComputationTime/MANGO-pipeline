"""Apply the upstream FLAb scatter style to the frozen heavy-domain analysis."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', 'artifacts/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flab_common import MODEL_LABELS
from plot_flab import plot_average_perplexity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(experiment_root, out_dir=None):
    root = Path(experiment_root)
    source = root / 'correlations'
    summary_path = source / 'local_summary.json'
    summary = json.loads(summary_path.read_text())
    if not summary['complete']:
        raise ValueError('The nine-model affinity analysis must be complete')
    out_dir = Path(out_dir) if out_dir else source / 'flab_style'
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = pd.read_csv(source / 'novel_heavy_pearson_by_model_target.csv').set_index('model')
    results = []; sources = {}
    for model in summary['models']:
        path = source / f'{model}_matched_pairs.csv.gz'
        sources[str(path)] = digest(path)
        frame = pd.read_csv(path)
        frame = frame.loc[frame.novel_heavy_eligible].copy()
        frame['alphaseq_affinity'] = frame.reliable_affinity
        label = {'esm3': 'ESM3', 'esm3_structure': 'ESM3+structure'}.get(model, MODEL_LABELS.get(model, model))
        for target in ('HER2', 'PD1', 'TIGIT'):
            output = out_dir / f'{target}_{model}.png'
            result = plot_average_perplexity(
                frame, target, model, output, label=f'{label} heavy',
                perplexity_column_name='heavy_perplexity',
            )
            np.testing.assert_allclose(result['pearson_r'], expected.loc[model, target], atol=1e-12, rtol=0)
            if result['n'] != summary['novel_heavy_pairs_by_target'][model][target]:
                raise ValueError(f'{model}/{target}: plotted cohort size changed')
            results.append(dict(result, figure=str(output), figure_sha256=digest(output)))
    pd.DataFrame(results).to_csv(out_dir / 'plot_correlations.csv', index=False)
    manifest = dict(complete=True, cohort='novel_heavy, reliable measurements',
        score='heavy-domain perplexity including EOS; no heavy/light averaging',
        affinity='published log10(Kd in nM), qualifying repeats averaged per normalized pair',
        training_overlap='HER2/PD1 light domains are training-seen; TIGIT is strict-unseen',
        analysis_summary_sha256=digest(summary_path), matched_scores_sha256=sources,
        implementation_sha256={str(p):digest(p) for p in
            (Path(__file__), Path(__file__).with_name('plot_flab.py'), Path(__file__).with_name('flab_common.py'))},
        figures=results)
    (out_dir / 'summary.json').write_text(json.dumps(manifest, indent=2) + '\n')
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<title>MANGO binding-affinity plots</title>',
        '<style>body{font:16px system-ui;margin:2rem}table{width:100%;border-collapse:collapse}'
        'th,td{padding:.5rem;border:1px solid #ddd}img{width:100%;max-width:420px}</style>',
        '<h1>MANGO binding-affinity plots</h1>',
        '<p>Quality-filtered, unseen-heavy-domain cohort. HER2 and PD1 light domains occur '
        'in training; TIGIT satisfies strict domain exclusion. Scores and correlations match '
        'the completed analysis.</p>',
        '<table><tr><th>Model</th><th>HER2</th><th>PD1</th><th>TIGIT</th></tr>']
    for model in summary['models']:
        gallery.append('<tr><th>' + html.escape(model) + '</th>')
        for target in ('HER2', 'PD1', 'TIGIT'):
            filename = html.escape(f'{target}_{model}.png')
            gallery.append(f'<td><a href="{filename}"><img src="{filename}" alt="{target} {html.escape(model)}"></a></td>')
        gallery.append('</tr>')
    gallery.append('</table></html>')
    (out_dir / 'index.html').write_text('\n'.join(gallery) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', default='artifacts/experiments/mango_binding_affinity')
    parser.add_argument('--out-dir')
    main(**vars(parser.parse_args()))
