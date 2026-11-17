"""Scientific pairing, parameter matching, region objectives and safe reporting."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from affinity_study_common import write_json,read_json
from short_affinity_common import balanced_select,schedule,affinity_pairs,ARMS


def row(i,split='train'):
    return dict(id=str(i),split=split,source_split=split,biological_group=f'g{i%4}',ab_ag_cluster=f'g{i%4}',
                resolved_ag_seq=f'antigen{i%4}',resolved_H_seq=f'heavy{i}',resolved_L_seq=f'light{i}',example_target_chains='heavy;light')


class DataTests(unittest.TestCase):
    def test_balanced_selection_preserves_groups_without_duplicates(self):
        rows=[row(i) for i in range(20)];selected=balanced_select(rows,8)
        self.assertEqual(len({r['id'] for r in selected}),8)
        self.assertEqual(Counter(r['biological_group'] for r in selected),dict(g0=2,g1=2,g2=2,g3=2))

    def test_schedule_preserves_both_selector_antigen_marginals(self):
        rows=[row(i) for i in range(12)];lookup={r['id']:r for r in rows};items=schedule(rows,32,0)
        for chain in ('heavy','light'):
            selected=[r for r in items if r['chain']==chain]
            self.assertEqual(Counter(lookup[r['id']]['resolved_ag_seq'] for r in selected),Counter(lookup[r['shuffled_id']]['resolved_ag_seq'] for r in selected))
            self.assertTrue(all(lookup[r['id']]['resolved_ag_seq']!=lookup[r['shuffled_id']]['resolved_ag_seq'] for r in selected))
        self.assertEqual(items,schedule(rows,32,0));self.assertNotEqual(items,schedule(rows,32,1))

    def test_all_training_contexts_and_decoys_stay_in_training(self):
        rows=[row(i) for i in range(12)]+[row(99,'test')]
        for item in schedule(rows,32,0):
            self.assertTrue(all(item[k]!='99' for k in ('id','shuffled_id','decoy_correct','decoy_shuffled')))

    def test_single_antigen_is_infeasible_not_silently_kept(self):
        rows=[dict(row(i),resolved_ag_seq='one') for i in range(4)]
        with self.assertRaises(ValueError):schedule(rows,32,0)

    def test_affinity_pairs_require_same_assay_and_confident_kd_order(self):
        rows=[dict(row(i),resolved_ag_seq='same') for i in range(4)]
        def measured(i,assay='one',mean=0.,operator='='):
            return dict(id=str(i),target_id='target',parent_id='parent',assay_id=assay,assay_provenance_url='https://example.test/assay',
                construct_format='scfv',measurement_type='Kd',operator=operator,log10_kd_nm=mean,log10_kd_sd=.1,replicate_count=3)
        records=[measured(0),measured(1,mean=2),measured(2,'other',mean=3),measured(3,mean=4,operator='<')]
        retained,pairs=affinity_pairs(records,rows)
        self.assertEqual([(p['strong'],p['weak']) for p in pairs],[('0','1')]);self.assertEqual(len(retained),3)
        records[1]['log10_kd_sd']=2
        self.assertEqual(affinity_pairs(records,rows)[1],[])

    def test_missing_provenance_is_not_affinity_supervision(self):
        with self.assertRaisesRegex(ValueError,'provenance'):affinity_pairs([dict(id='0')],[row(0)])

    def test_all_thirteen_original_and_two_capacity_arms_are_declared(self):
        self.assertEqual(len(ARMS),15)
        self.assertEqual(sum(v['test']==6 for v in ARMS.values()),4)


@unittest.skipUnless(importlib.util.find_spec('torch'),'use the isolated model environment')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from iglm_baseline import MODEL_CONFIG
        torch.set_num_threads(1)
        cls.cfg=dict(MODEL_CONFIG,n_layer=1,n_head=4,n_embd=32,n_inner=64,n_positions=128,n_ctx=128)

    def spec(self,arm):return dict(ARMS[arm],seed=3,id=f'{arm}_s3',arm=arm)

    def test_unconditional_parameter_counts_match_both_pooling_models(self):
        from short_affinity_model import build,model_parameters
        for conditional,unconditional in (('C','U'),('L','U_L')):
            a=build(self.spec(conditional),self.cfg);b=build(self.spec(unconditional),self.cfg)
            ratio=sum(p.numel() for p in model_parameters(a))/sum(p.numel() for p in model_parameters(b))
            self.assertLess(abs(ratio-1),.001)

    def test_zero_output_initialization_preserves_native_likelihood(self):
        import torch
        from short_affinity_model import build
        from controlled_antigen_model import mode
        from antigen_diagnostics import iglm_token_nll
        reference=build(self.spec('U_native'),self.cfg);mode(reference,False);seq='ACDEFGHIKLMNPQRSTVWY'
        for arm in ('C','L','U','U_L'):
            model=build(self.spec(arm),self.cfg);mode(model,False)
            for chain in ('heavy','light'):
                a=iglm_token_nll(reference,chain,seq);b=iglm_token_nll(model,chain,seq,torch.randn(1,43,1280) if model.branch else None)
                self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=0),arm)

    def test_learned_memory_has_fixed_size_and_receives_gradients(self):
        import torch
        from short_affinity_model import build
        model=build(self.spec('L'),self.cfg)
        ag=torch.randn(1,71,1280,requires_grad=True);memory=model.antigen_projection(ag)
        self.assertEqual(tuple(memory.shape),(1,32,32));memory.square().mean().backward()
        self.assertGreater(float(ag.grad.abs().sum()),0)
        self.assertGreater(float(model.antigen_projection.queries.grad.abs().sum()),0)

    def test_ranking_gradients_prefer_stronger_binding(self):
        import torch
        from short_affinity_model import preference
        a=torch.tensor([2.,2.],requires_grad=True);b=torch.tensor([2.,2.],requires_grad=True)
        preference(a,b).backward();self.assertTrue((a.grad>0).all());self.assertTrue((b.grad<0).all())

    def test_region_balancing_preserves_framework_and_eos(self):
        import torch
        from short_affinity_model import cdr_loss
        values=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        loss=cdr_loss(values,['framework','framework','cdr3','eos']);loss.backward()
        self.assertAlmostEqual(values.grad[2].item(),.5)
        self.assertTrue(all(values.grad[i]>0 for i in (0,1,3)))

    def test_learned_checkpoint_restores_eval_and_likelihood(self):
        import torch
        from short_affinity_model import build,save,load
        from controlled_antigen_model import mode,parity
        model=build(self.spec('L'),self.cfg)
        with torch.no_grad():model.lm.transformer.h[0].crossattention.c_proj.weight.fill_(.01)
        mode(model,False);ag=torch.randn(1,23,1280)
        for chain in ('heavy','light'):self.assertLess(parity(model,ag,['ACDEFGHIKLMNPQ','ACDEFGHIKLMN'],chain)['max_nll_error'],3e-6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';save(path,model,self.spec('L'),{})
            restored,_=load(path);self.assertFalse(restored.lm.training)
            for a,b in zip(model.lm.parameters(),restored.lm.parameters()):self.assertTrue(torch.equal(a,b))

    def test_windows_cover_every_residue_without_exceeding_budget(self):
        from short_affinity_embeddings import windows
        for length in (1,511,512,513,1024,2048):
            intervals=windows(length,512,128);covered=set()
            for a,b in intervals:self.assertLessEqual(b-a,512);covered.update(range(a,b))
            self.assertEqual(covered,set(range(length)))

    def test_deadline_cannot_silently_extend_fit(self):
        from short_affinity_worker import Deadline
        with self.assertRaises(TimeoutError):Deadline(0).check()


@unittest.skipUnless(importlib.util.find_spec('scipy') and importlib.util.find_spec('matplotlib'),'use data analysis environment')
class ReportingTests(unittest.TestCase):
    def test_infeasible_cohort_has_an_honest_complete_execution_report(self):
        from report_short_affinity import report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/'protocol.json',dict(settings=dict(seeds=[0,1,2],bootstrap_repeats=10,seed=0,nll_improvement=.01,nll_tolerance=.02,correlation_improvement=.03)))
            write_json(root/'cohort_audit.json',dict(feasible=False,reason='too_few_groups'))
            write_json(root/'ranking_audit.json',dict(feasible=False,reason='no_verified_assays'))
            write_json(root/'state.json',dict(tests={str(n):dict(status='infeasible') for n in range(1,7)},failed_stages=[],gpu_budget_used_seconds=0))
            report(root);result=read_json(root/'results_report.json')
            self.assertTrue(result['complete']);self.assertFalse(result['data_audit']['feasible']);self.assertEqual(result['runs'],[])
            self.assertFalse(any(v['confirmed'] for v in result['decisions'].values()))


if __name__=='__main__':unittest.main()
