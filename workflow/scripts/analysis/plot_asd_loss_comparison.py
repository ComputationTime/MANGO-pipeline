"""Compare ordinary NLL for scratch/ASD-pretrained CE/focal ESM2 models."""
import contextlib
import json
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


def plot(paths, figure, table, iglm_paths=None):
    values = [json.loads(Path(p).read_text()) for p in paths]
    expected = {(phase, loss, arch) for phase in ('sabdab2_scratch', 'sabdab2_finetuned')
                for loss in ('cross_entropy', 'focal') for arch in ('mango', 'cross_attention')}
    keys = [(p['experiment_phase'], p['objective'], p['architecture_label']) for p in values]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Exactly the eight requested final models are required')
    if len({p['cohort_sha256'] for p in values}) != 1:
        raise ValueError('All final models must share the native cohort')
    rows, counts = [], {}
    for p in values:
        if p.get('evaluation_loss') != 'cross_entropy' or p['embedding_tag'] != 'esm2':
            raise ValueError('Comparison requires ordinary NLL on frozen ESM2 inputs')
        transferred = p['experiment_phase'] == 'sabdab2_finetuned'
        if bool(p.get('initialization')) != transferred:
            raise ValueError('Scratch/transfer checkpoint provenance is inconsistent')
        for split in ('train', 'val', 'test_cath_assigned', 'test'):
            value = p['splits'][split]
            membership = (value['n_examples'], value['n_tokens'])
            if counts.setdefault(split, membership) != membership:
                raise ValueError('Final model evaluation memberships differ')
            rows.append(dict(phase=p['experiment_phase'], objective=p['objective'],
                architecture=p['architecture_label'], split=split, nll=value['nll'],
                perplexity=value['perplexity'], n_examples=value['n_examples'], n_tokens=value['n_tokens'],
                cohort_sha256=p['cohort_sha256'], run_id=p['run_id'],
                training_loss=json.dumps(p['training_loss'], sort_keys=True),
                initialization_sha256=(p.get('initialization') or {}).get('sha256', '')))
    iglm_values = [json.loads(Path(p).read_text()) for p in (iglm_paths or [])]
    if iglm_paths:
        phases = [p['experiment_phase'] for p in iglm_values]
        if len(phases) != 2 or set(phases) != {'sabdab2_scratch', 'sabdab2_finetuned'}:
            raise ValueError('Exactly two native IgLM training routes are required')
        for p in iglm_values:
            if (p['cohort_sha256'] != values[0]['cohort_sha256'] or
                    p.get('model_family') != 'IgLM' or p.get('antigen_conditioning') is not False or
                    p.get('public_pretrained_weights_loaded') is not False or
                    p.get('antibody_prompt_used') is not False or
                    p.get('target_chains') != ['heavy', 'light'] or
                    p.get('training_loss') != {'name': 'cross_entropy'} or
                    p.get('evaluation_loss') != 'cross_entropy' or
                    p.get('checkpoint_selection') != 'validation_nll' or
                    p.get('loss_reduction') != 'target_token_mean_including_eos'):
                raise ValueError('IgLM must match the cohort and ordinary NLL protocol')
            transferred = p['experiment_phase'] == 'sabdab2_finetuned'
            initial = p.get('transfer_initialization')
            if bool(initial) != transferred:
                raise ValueError('IgLM scratch/transfer checkpoint provenance is inconsistent')
            for split in ('train', 'val', 'test_cath_assigned', 'test'):
                value = p['splits'][split]
                if counts[split] != (value['n_examples'], value['n_tokens']):
                    raise ValueError('IgLM and MANGO evaluation memberships differ')
                for chain in ('heavy', 'light'):
                    reference = values[0]['splits'][split]['by_chain'][chain]
                    actual = value['by_chain'][chain]
                    if any(reference[key] != actual[key] for key in ('n_examples', 'n_tokens')):
                        raise ValueError('IgLM and MANGO per-chain memberships differ')
                rows.append(dict(phase=p['experiment_phase'], objective='cross_entropy',
                    architecture='iglm', split=split, nll=value['nll'], perplexity=value['perplexity'],
                    n_examples=value['n_examples'], n_tokens=value['n_tokens'],
                    cohort_sha256=p['cohort_sha256'], run_id=p['run_id'],
                    training_loss=json.dumps(p['training_loss'], sort_keys=True),
                    initialization_sha256=(initial or {}).get('sha256', '')))
    data = pd.DataFrame(rows)
    combinations = [('cross_entropy', 'mango'), ('focal', 'mango'),
                    ('cross_entropy', 'cross_attention'), ('focal', 'cross_attention')]
    labels = ['Normal\nMANGO\nStandard loss', 'Normal\nMANGO\nFocal loss (γ=2)',
              'Cross-attention\nMANGO\nStandard loss', 'Cross-attention\nMANGO\nFocal loss (γ=2)']
    if iglm_values:
        combinations.append(('cross_entropy', 'iglm'))
        labels.append('IgLM\n(no antigen)\nStandard loss')
    fig, axes = plt.subplots(1, 2, figsize=(18 if iglm_values else 16, 7.5), sharey=True)
    # Keep the secondary full-test results in the numerical table, while the
    # figure focuses on training and the primary CATH-assigned test subset.
    for ax, split, label in zip(axes, ('train', 'test_cath_assigned'),
                               ('Training NLL', 'Held-out test NLL')):
        for offset, phase, color, route in [
                (-.18, 'sabdab2_scratch', '#4C78A8', 'SAbDab2 only (trained from scratch)'),
                (.18, 'sabdab2_finetuned', '#F58518', 'ASD pretraining → SAbDab2 fine-tuning')]:
            scores = [float(data.loc[(data.phase == phase) & (data.objective == loss) &
                                    (data.architecture == arch) & (data.split == split), 'nll'].iloc[0])
                      for loss, arch in combinations]
            bars = ax.bar(np.arange(len(combinations)) + offset, scores, width=.36, color=color, label=route)
            pc.label_bars(ax, bars, scores)
        ax.set_xticks(np.arange(len(combinations)), labels)
        n_records = counts[split][0] // 2
        subset = ' · CATH-assigned subset' if split == 'test_cath_assigned' else ''
        pc.style(ax, f'{label}\n{n_records:,} SAbDab2 records{subset}',
                 ylabel='Antibody sequence NLL (nats/token)' if split == 'train' else '')
        ax.tick_params(axis='x', labelsize=10, pad=9)
        ax.margins(y=.12)
    title = 'Effect of ASD pretraining on antibody sequence prediction'
    subtitle = 'Standard loss = cross-entropy · Focal loss: γ=2 · Lower NLL is better'
    manifest_path = Path(paths[0]).parents[3] / 'experiment_manifest.json'
    if manifest_path.is_file():
        configuration = json.loads(manifest_path.read_text())['configuration']
        if configuration.get('prototype'):
            subtitle = ('HER2-free ASD pilot · One epoch per training stage · '
                        'Standard loss = cross-entropy · Lower NLL is better')
    if iglm_values:
        title = 'ASD pretraining comparison: MANGO and IgLM'
    fig.suptitle(title, y=.98, fontsize=18, fontweight='bold')
    fig.text(.5, .925, subtitle, ha='center', fontsize=11, color=pc.TEXT)
    handles, routes = axes[0].get_legend_handles_labels()
    fig.legend(handles, routes, loc='upper center', bbox_to_anchor=(.5, .885),
               ncols=2, frameon=False, fontsize=11, columnspacing=2.5)
    full_test_records = counts['test'][0] // 2
    caption = ('All bars show standard NLL over heavy/light residues + EOS, including focal-trained models.\n'
               f'The full {full_test_records:,}-record test results and validation scores remain in the accompanying CSV.')
    if iglm_values:
        caption += '\nMANGO: frozen ESM2 antigen inputs + chain type. IgLM: chain type + fixed species token; no antigen input.'
    fig.text(.5, .025, caption, ha='center', va='bottom', fontsize=10, color=pc.TEXT,
             linespacing=1.5)
    pc.normalize_typography(fig)
    # Reserve separate bands for the shared legend, panel titles, tick labels
    # and caption; the common tight-layout saver would collapse these bands.
    fig.subplots_adjust(left=.065, right=.99, bottom=.28, top=.74, wspace=.10)
    Path(table).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(table, index=False)
    Path(figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {figure} and {table}', flush=True)


if __name__ == '__main__':
    smk = globals()['snakemake']
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            plot(list(smk.input.evaluations), smk.output.figure, smk.output.data,
                 iglm_paths=list(getattr(smk.input, 'iglm_evaluations', [])))
        except Exception:
            traceback.print_exc()
            raise
