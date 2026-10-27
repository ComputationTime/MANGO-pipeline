"""Independently audit prepared cohorts and all completed transfer evaluations."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from chain_examples import example_chains
from prepare_abrank_transfer import assert_disjoint


def read_csv(path):
    with Path(path).open(newline='') as fh:
        return list(csv.DictReader(fh))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, cohort_only=False):
    root=Path(root);cohort=root/'cohort'
    audit=json.loads((cohort/'split_audit.json').read_text())
    source=audit.get('pretraining_dataset', 'abrank')
    for path,expected in audit['sha256'].items():
        if sha256(path)!=expected: raise ValueError(f'Pinned input or cohort changed: {path}')
    snapshot=json.loads((cohort/'input_manifest.json').read_text())
    for path,expected in snapshot['protected_figures'].items():
        if sha256(path)!=expected: raise ValueError(f'Original figure changed: {path}')
    native=read_csv(cohort/'sabdab2_records.csv')
    protected=read_csv(cohort/'protected_sabdab2_test.csv')
    retained=[{key:r[key] for key in protected[0]} for r in native if r['split']=='test']
    if retained!=protected: raise ValueError('The original SAbDab2 test changed')
    primary_ids={r['id'] for r in read_csv(cohort/'sabdab2_test_coverage.csv') if r['cath_assigned_all_antigen_chains']=='True'}
    findings=[]
    if source == 'asd':
        pretraining=read_csv(cohort/'asd_records.csv')
        assert_disjoint(pretraining, [r for r in native if r['split'] in {'val','test'}])
        seen=set()
        for row in pretraining:
            if audit.get('record_id_policy') == 'minimum_qualifying_frozen_ASD_row_id_v1':
                if row['id'] != f"asd_{int(row['source_row_id']):09d}":
                    raise ValueError('ASD control identity is not the pinned source row')
            for chain in example_chains(row):
                key=(row['resolved_ag_seq'],chain,row['resolved_H_seq' if chain=='heavy' else 'resolved_L_seq'])
                if key in seen: raise ValueError('Duplicate ASD individual-chain example')
                seen.add(key)
    for phase in [source,'sabdab2']:
        records=read_csv(cohort/f'{phase}_records.csv')
        by_split={split:[r for r in records if r['split']==split] for split in ['train','val','test']}
        if phase=='sabdab2': by_split['test_cath_assigned']=[r for r in by_split['test'] if r['id'] in primary_ids]
        for first,second in [('train','val'),('train','test'),('val','test')]:
            left,right=by_split[first],by_split[second]
            for field in ['resolved_H_seq','resolved_L_seq','ab_ag_cluster']:
                if {r[field] for r in left}&{r[field] for r in right}:
                    raise ValueError(f'{phase}: {field} crosses {first}/{second}')
            families=lambda rows:{f for r in rows for f in r['cath_families'].split(';') if f}
            if families(left)&families(right):raise ValueError('CATH family leakage')
        for path in ([] if cohort_only else sorted((root/phase/'runs').glob('*/eval.json'))):
            value=json.loads(path.read_text())
            directory=path.parent
            model=json.loads((directory/'model_config.json').read_text())
            runtime=json.loads((directory/'training_runtime.json').read_text())
            if runtime['status']!='complete':raise ValueError(f'Incomplete training: {path}')
            history=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if line]
            best=min(history,key=lambda r:r['val_loss'])
            if value['checkpoint_epoch']!=best['epoch'] or abs(value['checkpoint_val_loss']-best['val_loss'])>1e-10:
                raise ValueError('Checkpoint selection differs from validation history')
            if model['cohort_sha256']!=value['cohort_sha256']:raise ValueError('Model/evaluation cohorts differ')
            for split,rows in by_split.items():
                result=value['splits'][split]
                tokens={chain:sum(len(r[column])+1 for r in rows if chain in example_chains(r)) for chain,column in [('heavy','resolved_H_seq'),('light','resolved_L_seq')]}
                examples={chain:sum(chain in example_chains(r) for r in rows) for chain in tokens}
                if result['n_examples']!=sum(examples.values()) or result['n_tokens']!=sum(tokens.values()) or result['n_skipped_missing_embeddings']!=0:
                    raise ValueError(f'Evaluation membership/token counts differ: {path} {split}')
                for chain in ['heavy','light']:
                    if result['by_chain'][chain]['n_examples']!=examples[chain] or result['by_chain'][chain]['n_tokens']!=tokens[chain]:
                        raise ValueError('Chain counts differ')
                pooled=sum(result['by_chain'][c]['nll']*tokens[c] for c in tokens)/sum(tokens.values())
                if abs(pooled-result['nll'])>1e-10 or abs(math.exp(result['nll'])-result['perplexity'])>1e-10:
                    raise ValueError('NLL/perplexity aggregation differs')
            if phase=='sabdab2':
                init=value['initialization']
                source_checkpoint=root/source/'runs'/directory.name/'checkpoints'/'best.pt'
                source_model=json.loads((source_checkpoint.parent.parent/'model_config.json').read_text())
                if Path(init['path']).resolve()!=source_checkpoint.resolve() or init['sha256']!=sha256(source_checkpoint) or init['optimizer_loaded']:
                    raise ValueError('Wrong fine-tuning initialization')
                if init['source_cohort_sha256']!=source_model['cohort_sha256'] or source_model['architecture']!=model['architecture']:
                    raise ValueError('Incompatible transfer initialization')
            findings.append(dict(phase=phase,embedder=value['embedder'],passed=True))
    if cohort_only:
        result=dict(status='cohort_verified', pretraining_dataset=source,
                    primary_test_records=len(primary_ids), protected_test_records=len(protected),
                    original_figures_unchanged=True, all_pinned_hashes_match=True)
        (root/'cohort_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return result
    expected=json.loads((root/'experiment_manifest.json').read_text())['configuration']['controls']
    complete=len(findings)==2*len(expected)
    plot_checks={}
    for phase,filename in [(source,f'{source}_controls'),('sabdab2','sabdab2_finetuned_controls')]:
        figure=root/'plots'/f'{filename}.png'
        table=root/'plots'/f'{filename}.csv'
        if figure.is_file() and table.is_file():
            rows=read_csv(table)
            for row in rows:
                value=json.loads((root/phase/'runs'/row['embedder']/'eval.json').read_text())
                if abs(float(row['nll'])-value['splits'][row['split']]['nll'])>1e-10:
                    raise ValueError('Plotted NLL differs from evaluation')
            if len(rows)!=len(expected)*(3 if phase=='sabdab2' else 2):
                raise ValueError('Plot omits requested controls or splits')
            plot_checks[phase]={'figure_sha256':sha256(figure),'data_sha256':sha256(table),'rows':len(rows)}
    result=dict(status='passed' if complete and len(plot_checks)==2 else 'partial_verified',
                expected_evaluations=2*len(expected), completed_evaluations_checked=len(findings),evaluations=findings,
                primary_test_records=len(primary_ids),protected_test_records=len(protected),
                original_figures_unchanged=True,all_pinned_hashes_match=True,plots=plot_checks)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--cohort-only',action='store_true')
    args=parser.parse_args()
    verify(args.root,cohort_only=args.cohort_only)
