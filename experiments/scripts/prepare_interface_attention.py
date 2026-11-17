"""Freeze a new study while reusing previous antigen-only inputs read-only."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
import json
from pathlib import Path

from affinity_study_common import digest, freeze, read_csv, read_json, write_csv, write_json
from affinity_ranking_pilot import native_schedule


def main(config, settings):
    cfg = json.loads(settings)
    root = Path(cfg['experiment_root'])
    source = Path(cfg['source_root'])
    if root.resolve() == source.resolve():
        raise ValueError('New outputs must not overwrite the previous experiment')
    root.mkdir(parents=True, exist_ok=True)
    old = read_json(source/'protocol.json')
    prepared = read_json(source/'prepared.json')
    preserved = {}
    for field in ('inputs_sha256', 'implementation_sha256'):
        for path, expected in old[field].items():
            if digest(path) != expected:
                raise ValueError('Previous frozen source changed: ' + path)
            preserved[path] = expected
    for path in source.glob('runs/*/final.pt'):
        marker = read_json(path.parent/'complete.json')
        if digest(path) != marker['checkpoint_sha256']:
            raise ValueError('Previous checkpoint changed')
        preserved[str(path)] = marker['checkpoint_sha256']
    for path in (source/'protocol.json', source/'prepared.json', Path('run_gpu.sh')):
        preserved[str(path)] = digest(path)
    freeze(root/'preserved_previous.json', dict(files_sha256=preserved))
    rows = read_csv(source/'records.csv')
    pair = read_json(source/'ranking_pair.json')
    if cfg['smoke']:
        pairids = {pair[k]['id'] for k in ('strong', 'weak')}
        rows = ([r for r in rows if r['id'] in pairids]
                + [r for r in rows if r['split']=='train' and r['id'] not in pairids][:2]
                + [r for r in rows if r['split']=='test'][:2])
    for key in ('ab_ag_cluster', 'biological_group'):
        if {r[key] for r in rows if r['split']=='train'} & {r[key] for r in rows if r['split']=='test'}:
            raise ValueError('A protected group crosses splits')
    if any(r['source_split']!='train' for r in rows if r['split']=='train'):
        raise ValueError('A previous holdout entered training')
    implementation = [str(Path(__file__)), 'experiments/scripts/interface_attention_model.py',
                      'experiments/scripts/interface_attention_worker.py',
                      'experiments/scripts/report_interface_attention.py',
                      'experiments/scripts/validate_interface_attention.py',
                      'experiments/workflow/interface_attention.smk', 'run_interface_attention.sh',
                      'experiments/tests/test_interface_attention.py',
                      'experiments/scripts/interface_supervision_worker.py',
                      'experiments/scripts/interface_supervision_model.py',
                      'experiments/scripts/interface_supervision_common.py',
                      'experiments/scripts/affinity_study_common.py',
                      'experiments/scripts/affinity_ranking_pilot.py',
                      'experiments/scripts/controlled_antigen_model.py',
                      'workflow/scripts/iglm_baseline.py',
                      'workflow/scripts/device_common.py',
                      'experiments/scripts/short_affinity_worker.py']
    sources = [config, str(source/'protocol.json'), str(source/'prepared.json'),
               str(root/'preserved_previous.json')]
    freeze(root/'protocol.json', dict(
        settings=cfg, inputs_sha256={p:digest(p) for p in sources},
        implementation_sha256={p:digest(p) for p in implementation},
        primary='Paired change versus control in equal-target macro affinity Spearman of negative full-heavy NLL versus negative supplied log10 KD.',
        contrasts=['shared_tag minus control', 'contact_bias minus control', 'contact_bias minus shared_tag', 'each MANGO arm minus matched scratch IgLM'],
        mechanism='Group-averaged native within-residue attention AP, contact AP, attention mass enrichment on native contacts; NLL and affinity sensitivity to routing off/shuffled.',
        input='Frozen full-residue ESM2 plus canonical antigen-only geometry from interface_supervision_v1; native contacts are labels only.',
        prediction='Every layer predicts contacts of the NEXT residue from its pre-cross-attention state; no current/future target residue enters that prediction.',
        shared_tag='The same learned hidden-space vector, initially N(0,0.1), is added before Q/K projections, scaled by predicted paratope/epitope sigmoid membership. V is unchanged to isolate attention routing.',
        contact_bias='Add 2*sigmoid(predicted pair contact logit) to every attention head before softmax; fixed coefficient, no tuning.',
        supervision='All conditional arms receive the identical auxiliary loss and backbone gradients: layer-average [0.5 balanced pair BCE + 0.25 paratope BCE + 0.25 epitope BCE], weighted by 0.1.',
        intervention='Off removes only routing; shuffled deterministically permutes only routed antigen memberships/contact scores among real residues. Predictor supervision/sequence/memory/values remain aligned.',
        training='Five new paired seeds, identical initialization and group-balanced schedules; 1000 native updates plus 300 updates with the same RNase pair CE exposure. No affinity ranking loss or held-out model selection.',
        uncertainty='Sample SD across five paired optimization seeds. Additionally 95% native-group bootstrap intervals for mean paired native effects, conditional on these trained seeds. No biological affinity confidence interval.',
        limitations=['Small exploratory cohort: 99 training records/33 protection groups; 35 held-out heavy examples/16 groups.',
                     'Affinity panels were previously inspected; these are exploratory comparisons, not fresh confirmation.',
                     'Native geometry is experimental; external geometry is predicted antigen-alone Chai structure.',
                     'Region predictions may average over multiple possible interfaces for an antigen.',
                     'Contact localization and sequence likelihood need not measure quantitative binding strength.',
                     'IgLM matches total routing-model parameters; the control contains the same auxiliary parameters, unused in its generation path.']))
    write_csv(root/'records.csv', rows)
    write_json(root/'ranking_pair.json', pair)
    external = read_csv(source/'external_selection.csv')
    if cfg['smoke']:
        external = [r for t in ('HER2','PD1','TIGIT') for r in [x for x in external if x['target']==t][:8]]
    write_csv(root/'external_selection.csv', external)
    write_csv(root/'antigens.csv', read_csv(source/'antigens.csv'))
    for seed in cfg['seeds']:
        for phase, count in [('pretrain',cfg['pretraining_updates']), ('adapt',cfg['adaptation_updates'])]:
            write_json(root/'schedules'/f'{phase}_{seed}.json', dict(items=native_schedule(
                rows, count, cfg['native_batch_size'], seed+(0 if phase=='pretrain' else 100000))))
    names = ['records.csv','ranking_pair.json','external_selection.csv','antigens.csv']
    names += [str(p.relative_to(root)) for p in (root/'schedules').glob('*.json')]
    for field in ('inputs','labels'):
        for info in prepared[field].values():
            if digest(info['path']) != info['sha256']:
                raise ValueError('Previous antigen input/contact label changed')
    write_json(root/'prepared.json', dict(
        complete=True, inputs=prepared['inputs'], labels=prepared['labels'],
        files_sha256={n:digest(root/n) for n in names},
        counts={s:dict(records=sum(r['split']==s for r in rows),
                       groups=len({r['biological_group'] for r in rows if r['split']==s})) for s in ('train','test')},
        antibody_information_in_conditioning=False, previous_outputs_reused_read_only=True))
    print('Frozen interface-attention experiment:', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--settings-json', required=True)
    args = parser.parse_args()
    main(args.config, args.settings_json)
