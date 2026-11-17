"""Compare frozen models using held-out antibodies and measured assay labels."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from finalize_openalphaseq_pairing import digest, write_json

FAMILIES = ('RBD', 'TIGIT', 'PD1', 'HER2')
TIE_TOLERANCE = 3e-6  # nats/token; covers measured full-precision scorer drift.


def retrieval(scores, positives, tolerance=TIE_TOLERANCE):
    """Expected top-1/MRR under uniform ordering of tied scores, multilabel safe."""
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    valid = positives.any(axis=1) & np.isfinite(scores).all(axis=1)
    scores, positives = scores[valid], positives[valid]
    if not len(scores):
        return valid, np.zeros(0), np.zeros(0)
    best = np.where(positives, scores, np.inf).min(axis=1)
    better = (scores < best[:, None] - tolerance).sum(axis=1)
    tied = np.abs(scores - best[:, None]) <= tolerance
    count = tied.sum(axis=1)
    relevant = (tied & positives).sum(axis=1)
    top1 = np.where(better == 0, relevant / count, 0.)
    mrr = np.empty(len(scores))
    for b, t, p in set(zip(better.tolist(), count.tolist(), relevant.tolist())):
        value = sum(math.comb(t-r, p-1) / math.comb(t, p) / (b+r) for r in range(1, t-p+2))
        mrr[(better == b) & (count == t) & (relevant == p)] = value
    return valid, top1, mrr


def binary_auc(labels, scores):
    """Mann–Whitney AUROC with average ranks for exact ties."""
    labels = np.asarray(labels, dtype=bool); scores = np.asarray(scores, dtype=float)
    p = int(labels.sum()); n = len(labels) - p
    if not p or not n:
        return None
    order = np.argsort(scores, kind='stable'); sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts + ends + 1) / 2., ends - starts)
    return float((ranks[labels[order]].sum() - p * (p+1) / 2.) / (p*n))


def within_query_auc(scores, labels, mask):
    """Only explicitly measured positive/background-negative comparisons count."""
    eligible = mask & (labels == 1).any(axis=1) & (labels == -1).any(axis=1)
    scores, labels = scores[eligible], labels[eligible]
    n = len(scores); wins = np.zeros(n); pairs = np.zeros(n, dtype=int)
    for positive in range(scores.shape[1]):
        for negative in range(scores.shape[1]):
            selected = (labels[:, positive] == 1) & (labels[:, negative] == -1)
            delta = scores[:, negative] - scores[:, positive]
            wins[selected] += np.where(delta[selected] > TIE_TOLERANCE, 1.,
                                        np.where(delta[selected] < -TIE_TOLERANCE, 0., .5))
            pairs[selected] += 1
    selected = pairs > 0
    return dict(queries=int(selected.sum()), mean_auc=float(np.mean(wins[selected] / pairs[selected])) if selected.any() else None)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def specificity_control(scores, labels, mask, groups, seed=0, repeats=16):
    """Distinguish sequence-specific compatibility from fixed antigen preferences.

    Reassign scores to a different antibody from the same source target library.
    Source families describe experimental design and do not use binding labels.
    Random cyclic shifts have no fixed points; related library variants remain
    in the null, which is a diagnostic rather than a biological confidence bound.
    """
    groups = np.asarray(groups)
    valid = mask & np.isfinite(scores).all(axis=1)
    eligible = valid & (labels == 1).any(axis=1) & (labels == -1).any(axis=1)
    indices = np.flatnonzero(eligible); selected_labels = labels[indices]
    if not len(indices):
        return dict(queries=0, observed_auc=None, global_mean_auc=None, library_mean_auc=None,
                    reassigned_mean_auc=None, reassigned_sd=None, excess_over_reassignment=None, repeats=repeats)
    selected_mask = np.ones(len(indices), dtype=bool)
    observed = within_query_auc(scores[indices], selected_labels, selected_mask)['mean_auc']
    global_mean = np.broadcast_to(scores[valid].mean(axis=0), (len(indices),scores.shape[1]))
    library_mean = np.empty_like(global_mean); pools = []
    for group in sorted(set(groups[eligible])):
        pool = np.flatnonzero(valid & (groups == group))
        cases = np.flatnonzero(groups[indices] == group)
        library_mean[cases] = scores[pool].mean(axis=0)
        if len(pool) < 2:
            raise ValueError('Within-library reassignment requires at least two antibodies')
        pools.append((pool, cases, np.searchsorted(pool, indices[cases])))
    generator = np.random.default_rng(seed); null = []
    for _ in range(repeats):
        shuffled = np.empty_like(library_mean)
        for pool, cases, positions in pools:
            shift = generator.integers(1, len(pool))
            shuffled[cases] = scores[pool[(positions+shift) % len(pool)]]
        null.append(within_query_auc(shuffled, selected_labels, selected_mask)['mean_auc'])
    return dict(queries=len(indices), observed_auc=observed,
        global_mean_auc=within_query_auc(global_mean, selected_labels, selected_mask)['mean_auc'],
        library_mean_auc=within_query_auc(library_mean, selected_labels, selected_mask)['mean_auc'],
        reassigned_mean_auc=float(np.mean(null)), reassigned_sd=float(np.std(null)),
        excess_over_reassignment=float(observed-np.mean(null)), repeats=repeats)


def evaluate_matrix(scores, labels, families, mask):
    positives = labels == 1
    selected = mask & positives.any(axis=1) & np.isfinite(scores).all(axis=1)
    _, top1, mrr = retrieval(scores[selected], positives[selected])
    per_family = {}
    family_scores = np.column_stack([scores[:, families == f].mean(axis=1) for f in FAMILIES])
    family_positives = np.column_stack([positives[:, families == f].any(axis=1) for f in FAMILIES])
    _, family_top1, _ = retrieval(family_scores[selected], family_positives[selected])
    for j, family in enumerate(FAMILIES):
        # Exclude cross-family multi-positive cases from the single-family macro.
        members = family_positives[selected, j] & (family_positives[selected].sum(axis=1) == 1)
        per_family[family] = dict(positive_queries=int(members.sum()),
            protein_top1=float(top1[members].mean()) if members.any() else None,
            family_top1=float(family_top1[members].mean()) if members.any() else None,
            **{'measured_within_query_'+k:v for k,v in within_query_auc(scores, labels,
                mask & family_positives[:,j] & np.isfinite(scores).all(axis=1)).items()})
    macro = [r['family_top1'] for r in per_family.values() if r['family_top1'] is not None]
    return dict(positive_queries=int(selected.sum()), protein_top1=float(top1.mean()) if len(top1) else None,
        protein_mrr=float(mrr.mean()) if len(mrr) else None,
        family_balanced_top1=float(np.mean(macro)) if macro else None,
        families_present=len(macro), per_family=per_family,
        measured_within_query=within_query_auc(scores, labels, mask & np.isfinite(scores).all(axis=1)))


def report(study, partial=False):
    root = Path(study['experiment_root']); output = root / 'partial' if partial else root
    output.mkdir(parents=True, exist_ok=True)
    if not partial and not (root / 'scoring_complete.json').is_file():
        raise ValueError('Full scoring is incomplete; use --partial for a clearly marked interim report')
    manifest = json.loads((root / 'scoring_manifest.json').read_text())
    if manifest['configuration'] != study:
        raise ValueError('Scoring configuration differs from report configuration')
    audit = json.loads((root / 'cohort_audit.json').read_text())
    if (audit['cohort_sha256'] != digest(root / 'cohort.sqlite') or
        audit['training_records_sha256'] != digest(study['training_records']) or
        audit['raw_manifest_sha256'] != manifest['raw_manifest_sha256'] or
        manifest['raw_manifest_sha256'] != digest(root / 'raw_manifest.json')):
        raise ValueError('Held-out cohort or scoring source provenance changed')
    if not partial:
        completed = json.loads((root / 'scoring_complete.json').read_text())
        if (not completed['complete'] or completed['scoring_manifest_sha256'] != digest(root / 'scoring_manifest.json') or
            set(completed['models']) != set(study['models'])):
            raise ValueError('Scoring completion marker does not match this comparison')
    antigen_ids = manifest['antigen_order']; antigen_index = {key:i for i,key in enumerate(antigen_ids)}
    with sqlite3.connect(root / 'raw.sqlite') as db:
        antigen_info = {r[0]:dict(sequence=r[1], family=r[2], aliases=json.loads(r[3])) for r in db.execute('SELECT * FROM antigens')}
    families = np.array([antigen_info[key]['family'] for key in antigen_ids])
    with sqlite3.connect(root / 'cohort.sqlite') as db:
        query_rows = db.execute('SELECT id,heavy,light,heavy_novel,light_novel,strict_unseen,families FROM queries ORDER BY id').fetchall()
        targets = {r[0]:dict(chain=r[1], sequence=r[2], ntokens=r[4], ncdr3=r[5], seen=bool(r[6]))
                   for r in db.execute('SELECT * FROM targets')}
        label_rows = db.execute('SELECT query_id,antigen_id,label FROM labels').fetchall()
    scoredb = sqlite3.connect(f'file:{(root / "scores.sqlite").resolve()}?mode=ro', uri=True)
    scored = {name:{r[0] for r in scoredb.execute('SELECT target_id FROM scores WHERE model=?', (name,))}
              for name in study['models']}
    common = set.intersection(*scored.values())
    available = [q for q in query_rows if all(t in common for t,new in ((q[1],q[3]),(q[2],q[4])) if new)]
    if not partial and len(available) != len(query_rows):
        raise ValueError('Some held-out queries have missing model scores')
    query_rows = available; n = len(query_rows)
    if not n:
        raise ValueError('No complete held-out scores available')
    ids = np.array([q[0] for q in query_rows]); qindex = {key:i for i,key in enumerate(ids)}
    labels = np.zeros((n, len(antigen_ids)), dtype=np.int8)
    for qid, aid, label in label_rows:
        if qid in qindex:
            labels[qindex[qid],antigen_index[aid]] = {'positive':1, 'negative':-1, 'uncertain':0}[label]
    masks = dict(heavy_novel=np.array([bool(q[3]) for q in query_rows]),
                 novel_domains=np.ones(n, dtype=bool), strict_unseen=np.array([bool(q[5]) for q in query_rows]))
    matrices, rows, per_family_rows, binder_rows, control_rows = {}, [], [], [], []
    positive_nll = {}
    for name in study['models']:
        data = {r[0]:(np.frombuffer(r[1], dtype=np.float64).copy(), np.frombuffer(r[2], dtype=np.float64).copy())
                for r in scoredb.execute('SELECT target_id,loss_sum,cdr3_sum FROM scores WHERE model=?', (name,))}
        arrays = {key:np.full(labels.shape, np.nan) for key in ('heavy_nll','heavy_cdr3_nll','novel_nll','novel_cdr3_nll')}
        for i, q in enumerate(query_rows):
            parts = [t for t,new in ((q[1],q[3]),(q[2],q[4])) if new]
            assert parts and not any(targets[t]['seen'] for t in parts)
            arrays['novel_nll'][i] = sum((data[t][0] for t in parts)) / sum(targets[t]['ntokens'] for t in parts)
            cdr_count = sum(targets[t]['ncdr3'] for t in parts)
            if cdr_count:
                arrays['novel_cdr3_nll'][i] = sum((data[t][1] for t in parts)) / cdr_count
            if q[3]:
                arrays['heavy_nll'][i] = data[q[1]][0] / targets[q[1]]['ntokens']
                if targets[q[1]]['ncdr3']:
                    arrays['heavy_cdr3_nll'][i] = data[q[1]][1] / targets[q[1]]['ncdr3']
        del data
        matrices[name] = arrays
        scorepath = output / f'{name}_scores.npz'
        np.savez_compressed(scorepath, query_ids=ids, antigen_ids=np.array(antigen_ids), labels=labels,
                            heavy_novel=masks['heavy_novel'], strict_unseen=masks['strict_unseen'], **arrays)
        for cohort, mask in masks.items():
            for region in ('full', 'cdr3'):
                key = ('heavy' if cohort == 'heavy_novel' else 'novel') + ('_nll' if region == 'full' else '_cdr3_nll')
                raw = arrays[key]
                for calibration in ('raw', 'column_centered'):
                    # Label-free transductive centering; raw scores remain primary.
                    values = raw if calibration == 'raw' else raw - np.nanmean(raw[mask], axis=0)
                    result = evaluate_matrix(values, labels, families, mask)
                    base = dict(model=name, cohort=cohort, region=region, calibration=calibration)
                    rows.append(dict(**base, positive_queries=result['positive_queries'],
                        protein_top1=result['protein_top1'], protein_mrr=result['protein_mrr'],
                        family_balanced_top1=result['family_balanced_top1'], families_present=result['families_present'],
                        measured_within_query_auc=result['measured_within_query']['mean_auc'],
                        measured_within_query_count=result['measured_within_query']['queries']))
                    for family, values_by_family in result['per_family'].items():
                        per_family_rows.append(dict(**base, family=family, **values_by_family))
        pos = (labels == 1) & masks['heavy_novel'][:,None]
        positive_nll[name] = float(arrays['heavy_nll'][pos].mean()) if pos.any() else None
        print(f'{name}: wrote {n:,} queries × {len(antigen_ids)} proteins', flush=True)
    scoredb.close()
    source_groups = np.array([q[6] for q in query_rows])
    for name, arrays in matrices.items():
        for region, key in [('full','heavy_nll'), ('cdr3','heavy_cdr3_nll')]:
            control_rows.append(dict(model=name, region=region, source_family='all',
                **specificity_control(arrays[key], labels, masks['heavy_novel'], source_groups, study['seed'])))
            for family in FAMILIES:
                group_mask = np.array([family in json.loads(group) for group in source_groups])
                control_rows.append(dict(model=name, region=region, source_family=family,
                    **specificity_control(arrays[key], labels, masks['heavy_novel'] & group_mask, source_groups, study['seed'])))
    # Within each protein, classify only explicitly observed positive/negative pairs.
    for name, arrays in matrices.items():
        for j, aid in enumerate(antigen_ids):
            selected = masks['heavy_novel'] & (labels[:,j] != 0) & np.isfinite(arrays['heavy_nll'][:,j])
            y = labels[selected,j] == 1
            for method in ('negative_nll', 'iglm_conditioning_gain'):
                scores = (-arrays['heavy_nll'][selected,j] if method == 'negative_nll' else
                          matrices['iglm']['heavy_nll'][selected,j] - arrays['heavy_nll'][selected,j])
                binder_rows.append(dict(model=name, antigen_id=aid, antigen=';'.join(antigen_info[aid]['aliases']),
                    family=families[j], score=method, positives=int(y.sum()), negatives=int((~y).sum()),
                    auroc=binary_auc(y,scores)))
    primary = [r for r in rows if r['cohort']=='heavy_novel' and r['region']=='full' and r['calibration']=='raw']
    ranking = []
    for r in primary:
        by_family = []
        for family in FAMILIES:
            aucs = [a['auroc'] for a in binder_rows if a['model']==r['model'] and a['family']==family
                    and a['score']=='negative_nll' and a['auroc'] is not None]
            if aucs:
                by_family.append(float(np.mean(aucs)))
        control = next(c for c in control_rows if c['model']==r['model'] and c['region']=='full' and c['source_family']=='all')
        ranking.append(dict(**r, observed_positive_nll=positive_nll[r['model']],
            family_balanced_binder_auroc=float(np.mean(by_family)) if by_family else None,
            within_library_reassigned_auc=control['reassigned_mean_auc'],
            excess_within_query_auc=control['excess_over_reassignment']))
    ranking.sort(key=lambda r:-(r['family_balanced_top1'] if r['family_balanced_top1'] is not None else -1))
    write_csv(output / 'metrics.csv', rows); write_csv(output / 'per_family_metrics.csv', per_family_rows)
    write_csv(output / 'binder_metrics.csv', binder_rows); write_csv(output / 'rankings.csv', ranking)
    write_csv(output / 'specificity_controls.csv', control_rows)
    full_controls = [c for c in control_rows if c['region']=='full' and c['source_family']=='all']
    measured_winner = max(full_controls, key=lambda c:c['observed_auc'] if c['observed_auc'] is not None else -1)
    specificity_winner = max([c for c in full_controls if c['model']!='iglm'],
                            key=lambda c:c['excess_over_reassignment'] if c['excess_over_reassignment'] is not None else -1)
    summary = dict(complete=not partial, partial=partial, scored_queries=n,
        total_heldout_queries=audit['unique_queries'],
        candidate_antigens=len(antigen_ids), candidate_families=list(FAMILIES),
        primary_cohort='heavy_novel', primary_metric='family_balanced_top1',
        family_reduction='mean_NLL_across_panel_proteins_in_each_family',
        precision=manifest['precision'], tie_tolerance_nats_per_token=TIE_TOLERANCE,
        rankings=ranking, best_model=ranking[0]['model'],
        best_model_by_measured_pair_auc=measured_winner['model'],
        best_model_by_excess_over_reassignment=specificity_winner['model'],
        measured_pair_specificity_controls=full_controls,
        limitations=[
            'Exact antibody domains and complete training-domain substrings are excluded; this is not a lineage-disjoint evaluation.',
            'Optimization libraries contain many related variants of few parent antibodies; rows are not independent biological replicates.',
            'Unmeasured cross-protein pairs are unknown alternatives, not experimentally established nonbinders.',
            'Protein retrieval credits any reliable positive partner, including multiple RBD variants.',
            'Raw scores are primary. Column centering uses all evaluated queries without labels and is transductive.',
            'IgLM has no antigen input and receives expected chance credit for tied antigen scores.',
            'A positive/background-negative conflict is positive whenever any reliable binding observation exists.',
            'Full-chain NLL includes EOS; IMGT CDR3 NLL excludes framework and EOS.',
            'One epoch, one training seed, 16k ASD records; no models were retrained for this evaluation.'],
        scoring_manifest_sha256=digest(root / 'scoring_manifest.json'),
        cohort_audit_sha256=digest(root / 'cohort_audit.json'), implementation_sha256=digest(__file__))
    write_json(output / 'summary.json', summary)
    plot_results(output, ranking, per_family_rows)
    findings = ['# OpenAlphaSeq external pairing evaluation', '',
        f'**{"INTERIM — " if partial else ""}{n:,} held-out antibody queries; {len(antigen_ids)} candidate proteins.**', '',
        'Primary results use novel heavy domains. Each query is scored against every panel protein. '
        'Family scores average NLL over the proteins in that family; accuracy is averaged equally across the four target families.', '',
        '| Model | Family-balanced top-1 | Protein top-1 | Protein MRR | Binder AUROC (family-balanced) |',
        '|---|---:|---:|---:|---:|']
    for r in ranking:
        findings.append(f'| {r["model"]} | {r["family_balanced_top1"]:.3%} | {r["protein_top1"]:.3%} | '
                        f'{r["protein_mrr"]:.4f} | {r["family_balanced_binder_auroc"]:.4f} |')
    findings += ['', 'Binder AUROC uses only experimentally observed positive/background-negative pairs within each protein. '
                 'It measures a different property from choosing an annotated antigen partner.', '',
                 'Sequence-specificity controls reassign scores to a different antibody from the same source library. '
                 'A strong measured-pair AUROC that also occurs after reassignment can arise from a fixed antigen preference. '
                 'These controls use no binding labels to choose the donor antibody.', '',
                 '| Model | Measured-pair AUROC | After antibody reassignment | Excess |',
                 '|---|---:|---:|---:|---:|']
    for r in ranking:
        findings.append(f'| {r["model"]} | {r["measured_within_query_auc"]:.4f} | '
                        f'{r["within_library_reassigned_auc"]:.4f} | {r["excess_within_query_auc"]:+.4f} |')
    findings += ['', f'Highest raw family accuracy: **{ranking[0]["model"]}**. '
                 f'Highest measured-pair AUROC: **{measured_winner["model"]}**. '
                 f'Largest excess over reassignment: **{specificity_winner["model"]}**. '
                 'These are observed rankings within this panel, not evidence of generalization to independent antibody lineages.', '',
                 'Strict all-domains-unseen and novel-domain-composite results, CDR3 results, '
                 'and label-free column-centered diagnostics are in `metrics.csv`.', '']
    findings += ['- ' + text for text in summary['limitations']]
    (output / 'findings.md').write_text('\n'.join(findings) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def plot_results(output, ranking, family_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [r['model'] for r in ranking]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
    axes[0].bar(names, [100*r['family_balanced_top1'] for r in ranking])
    axes[0].axhline(25, color='black', ls=':', label='IgLM / tied chance')
    axes[0].set(ylabel='Family-balanced top-1 (%)', title='Antigen-family pairing, novel heavy chains', ylim=(0,100))
    axes[0].legend(fontsize=8)
    axes[1].bar(names, [r['measured_within_query_auc'] for r in ranking])
    axes[1].scatter(names, [r['within_library_reassigned_auc'] for r in ranking], color='darkorange',
                    marker='D', s=35, zorder=3, label='Reassigned within library')
    axes[1].axhline(.5, color='black', ls=':', label='IgLM / tied chance')
    axes[1].set(ylabel='Mean within-antibody AUROC', ylim=(0,1),
                title=f'Measured positive vs background-negative\n{ranking[0]["measured_within_query_count"]:,} antibodies')
    axes[1].legend(fontsize=8)
    x = np.arange(len(names)); width = .19
    for j, family in enumerate(FAMILIES):
        values = [next(r['family_top1'] for r in family_rows if r['model']==name and r['family']==family
                  and r['cohort']=='heavy_novel' and r['region']=='full' and r['calibration']=='raw') for name in names]
        axes[2].bar(x+(j-1.5)*width, [100*v if v is not None else np.nan for v in values], width, label=family)
    axes[2].set(xticks=x, xticklabels=names, ylabel='Correct family (%)', ylim=(0,100), title='Per target family')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis='x', rotation=25)
    fig.suptitle(('INTERIM — ' if output.name=='partial' else '') +
                 f'OpenAlphaSeq: {ranking[0]["positive_queries"]:,} novel-heavy antibodies with measured positive partners', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(output / 'pairing_comparison.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/openalphaseq_pairing.yaml')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if Path(args.config).suffix == '.json':
        study = json.loads(Path(args.config).read_text())
    else:
        import yaml
        study = yaml.safe_load(Path(args.config).read_text())
    report(study, partial=args.partial)
