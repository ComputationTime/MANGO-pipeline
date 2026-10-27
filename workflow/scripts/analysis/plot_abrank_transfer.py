"""Separate AbRank-only and AbRank-to-SAbDab2 conditioning-control figures."""
import json
import contextlib
import traceback
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


def plot(evaluations, phase, tags, labels, out_figure, out_data, audit, dpi=180):
    payloads = [json.loads(Path(p).read_text()) for p in evaluations]
    by_tag = {p['embedder']: p for p in payloads}
    if len(payloads) != len(by_tag) or set(by_tag) != set(tags):
        raise ValueError('Exactly one evaluation per requested control is required')
    if len({p['cohort_sha256'] for p in payloads}) != 1:
        raise ValueError('Cannot plot controls with different cohorts')
    pairing_path = Path(evaluations[0]).parents[2]/'embeddings'/'antigen'/'control_shuffled'/'biological_pairing_audit.json'
    pairing = json.loads(pairing_path.read_text()) if pairing_path.is_file() else {}
    definitions = [('train', 'Train'), ('test', 'Test')]
    source = audit.get('pretraining_dataset', 'abrank')
    source_label = 'ASD 2.0' if source == 'asd' else 'AbRank'
    title = f'{source_label} only: antigen conditioning controls'
    if phase == 'sabdab2':
        definitions = [('train', 'Train'), ('test_cath_assigned', 'Test: CATH assigned (primary)'),
                       ('test', 'Test: all records (secondary)')]
        title = f'{source_label} pretraining → SAbDab2 fine-tuning: conditioning controls'
        if any(not p.get('initialization') for p in payloads):
            raise ValueError('Every fine-tuned model must record its source checkpoint')
    rows = []
    counts = {}
    record_counts = {}
    for tag in tags:
        p = by_tag[tag]
        for split, label in definitions:
            value = p['splits'][split]
            pair = (value['n_examples'], value['n_tokens'])
            if split in counts and counts[split] != pair:
                raise ValueError('Control evaluation subsets differ')
            counts[split] = pair
            record_counts[split] = value.get('antigen_context_tokens', {}).get('n_records', value['n_examples']//2)
            rows.append(dict(phase=phase, embedder=tag, label=labels[tag], split=split,
                nll=value['nll'], perplexity=value['perplexity'], n_examples=value['n_examples'],
                n_tokens=value['n_tokens'], run_id=p['run_id'], cohort_sha256=p['cohort_sha256'],
                architecture=p['architecture'], checkpoint_epoch=p['checkpoint_epoch'],
                embedding_width=p.get('d_ag'),
                antigen_in_decoder_prefix=p.get('antigen_in_decoder_prefix'),
                cross_attention_trainable_parameters=p.get('cross_attention_trainable_parameters'),
                antigen_tokens_min=value.get('antigen_context_tokens', {}).get('min'),
                antigen_tokens_mean=value.get('antigen_context_tokens', {}).get('mean'),
                antigen_tokens_max=value.get('antigen_context_tokens', {}).get('max'),
                shuffled_same_antigen_fraction=(pairing.get(split, {}).get('same_antigen_sequence', 0) /
                    pairing[split]['records']) if tag == 'control_shuffled' and split in pairing else '',
                initialization_sha256=(p.get('initialization') or {}).get('sha256', '')))
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(17, 6.5))
    x = np.arange(len(tags))
    width = .8 / len(definitions)
    colors = pc.colors(tags)
    styles = [(None, 1.), ('///', .65), ('...', .35)]
    for i, ((split, label), (hatch, alpha)) in enumerate(zip(definitions, styles)):
        values = [data.loc[(data.embedder == tag) & (data.split == split), 'nll'].iloc[0] for tag in tags]
        bars = ax.bar(x + (i - (len(definitions)-1)/2)*width, values, width,
                      color=[colors[t] for t in tags], edgecolor=pc.TEXT, linewidth=.7,
                      hatch=hatch, alpha=alpha,
                      label=f'{label}: {record_counts[split]:,} records')
        pc.label_bars(ax, bars, values)
    ax.set_xticks(x, [labels[t].replace(' (', '\n(').replace(' ESM2', '\nESM2') for t in tags])
    pc.style(ax, title, xlabel='Conditioning representation', ylabel='Pooled heavy/light NLL (nats/token)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.17), ncols=len(definitions))
    footer = 'CATH family-disjoint splits; heavy/light variable domains; best validation checkpoint.\n'
    if phase != 'sabdab2':
        footer += ' Training is dominated by HER2; held-out antigen diversity is limited.'
    else:
        footer += f" {audit['secondary_only_test_records']} test records lack confident CATH assignment."
    if pairing.get('train', {}).get('records'):
        fraction = pairing['train']['same_antigen_sequence'] / pairing['train']['records']
        footer += f' Training record shuffle retains {fraction:.0%} identical antigens.'
    fig.text(.5, .01, footer, ha='center', fontsize=9, color=pc.TEXT)
    fig.subplots_adjust(bottom=.3)
    pc.save(fig, data, out_figure, out_data, dpi)


if __name__ == '__main__':
    smk = globals().get('snakemake')
    if smk is None:
        raise RuntimeError('Run through workflow/abrank_transfer.smk')
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            plot(list(smk.input.evaluations), smk.params.phase, list(smk.params.tags), dict(smk.params.labels),
                 smk.output.figure, smk.output.data, json.loads(Path(smk.input.audit).read_text()))
        except Exception:
            traceback.print_exc()
            raise
