"""Scientific controls, biological independence, and exact likelihood checks."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow/scripts'))
from affinity_study_common import fingerprint,write_json
from controlled_antigen_common import PRIMARY,SECONDARY,crossed_interval,score_arrays
from prepare_controlled_antigen import components,allocate


def record(name,split='train',family='',heavy=None,light=''):
    return dict(id=name,source_split=split,split=split,ab_ag_cluster=name,
                cath_families=family,resolved_H_seq=heavy or name,resolved_L_seq=light,
                resolved_ag_seq='antigen_'+name,example_target_chains='heavy')


class CohortTests(unittest.TestCase):
    def test_components_join_source_groups_families_and_near_lineages(self):
        rows=[record('a',family='family1'),record('b',family='family1'),record('c'),record('d')]
        rows[3]['ab_ag_cluster']='c'
        hits=[(fingerprint('b'),fingerprint('c'))]
        self.assertEqual(len(components(rows,hits)),1)

    def test_missing_light_chain_does_not_connect_unrelated_antibodies(self):
        self.assertEqual(len(components([record('a'),record('b')],[])),2)

    def test_exact_antigen_identity_also_prevents_cross_split_leakage(self):
        rows=[record('a'),record('b')];rows[1]['resolved_ag_seq']=rows[0]['resolved_ag_seq']
        self.assertEqual(len(components(rows,[])),1)

    def test_reserving_training_components_never_trains_old_holdouts(self):
        groups=[[record(str(i),'val' if i==0 else 'test' if i==1 else 'train')] for i in range(10)]
        settings=dict(minimum_groups=2,minimum_heavy_examples=2,minimum_training_records=4,minimum_training_antigens=4)
        rows=allocate(groups,settings)
        self.assertIsNotNone(rows)
        self.assertTrue(all(r['source_split']=='train' for r in rows if r['split']=='train'))
        self.assertEqual(len({r['biological_group'] for r in rows if r['split']=='val'}),2)
        self.assertEqual(len({r['biological_group'] for r in rows if r['split']=='test'}),2)
        self.assertIsNone(allocate(groups,dict(settings,minimum_groups=4)))

    def test_training_members_of_a_holdout_component_are_reserved_together(self):
        linked=[record('a','test'),record('b')]
        groups=[linked]+[[record(str(i))] for i in range(6)]
        settings=dict(minimum_groups=1,minimum_heavy_examples=1,minimum_training_records=3,minimum_training_antigens=3)
        rows=allocate(groups,settings);pair=[r for r in rows if r['id'] in ('a','b')]
        self.assertEqual(len({r['split'] for r in pair}),1)
        self.assertNotEqual(pair[0]['split'],'train')


class InferenceTests(unittest.TestCase):
    def test_token_weighting_and_seed_axis_are_preserved(self):
        result=crossed_interval([[1.,9.],[2.,18.]],[[100.,300.],[100.,300.]],500,0)
        self.assertAlmostEqual(result['estimate'],.0375)
        self.assertEqual(result['seed_points'],[.025,.05])
        self.assertEqual((result['seeds'],result['groups']),(2,2))
        self.assertLess(result['low'],result['estimate']);self.assertGreater(result['high'],result['estimate'])

    def test_identical_paired_models_have_zero_interval_and_no_significance(self):
        result=crossed_interval(np.zeros((3,20)),np.ones((3,20))*20,500,0)
        self.assertEqual((result['estimate'],result['low'],result['high'],result['pvalue']),(0.,0.,0.,1.))
        smoke=crossed_interval([[2.,4.]],[[1.,1.]],20)
        self.assertIsNone(smoke['low']);self.assertEqual(smoke['seed_points'],[3.])

    def test_pairing_requires_identical_targets_groups_and_tokens(self):
        def score(name,n=10,loss=20.):
            return dict(id=name,chain='heavy',variant='correct',group=name,loss_sum=loss,ntokens=n)
        tables={0:dict(correct=[score('a'),score('b')],shuffled=[score('a',loss=21.),score('b',loss=22.)])}
        losses,tokens=score_arrays(tables,PRIMARY['pairing'])
        self.assertEqual(losses.tolist(),[[1.,2.]]);self.assertEqual(tokens.tolist(),[[10,10]])
        tables[0]['shuffled'][1]['ntokens']=11
        with self.assertRaisesRegex(ValueError,'token mismatch'):score_arrays(tables,PRIMARY['pairing'])
        tables[0]['shuffled']=[score('a')]
        with self.assertRaisesRegex(ValueError,'Unpaired'):score_arrays(tables,PRIMARY['pairing'])

    def test_interaction_sign_measures_larger_pairing_benefit_when_frozen(self):
        values=dict(frozen_correct=1.,frozen_shuffled=1.2,joint_correct=1.1,joint_shuffled=1.15)
        self.assertAlmostEqual(sum(c*values[a] for a,c in SECONDARY['interaction']),.15)

    def test_infeasible_audit_still_finishes_amendment_and_report(self):
        from run_controlled_antigen import Controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference';reference.mkdir()
            write_json(reference/'selection.json',dict(external_runs=['old_s0','old_s1']))
            write_json(root/'cohort_audit.json',dict(feasible=False,reason='too_few_groups'))
            controller=Controller.__new__(Controller);controller.root=root
            controller.settings=dict(reference_root=str(reference));controller.jobs={};controller.failed=[];controller.decisions={}
            stages=[]
            with patch.object(controller,'stage',side_effect=lambda n,t:stages.append(n) or True):
                self.assertEqual(controller.run(),0)
            self.assertEqual(stages,['amendment_old_s0','amendment_old_s1','audit','report'])
            self.assertEqual(controller.decisions['data']['status'],'infeasible')

    def test_confirmation_is_gated_and_support_requires_corrected_evidence(self):
        from run_controlled_antigen import Controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference';reference.mkdir()
            write_json(reference/'selection.json',dict(external_runs=[]))
            write_json(root/'cohort_audit.json',dict(feasible=True))
            controller=Controller.__new__(Controller);controller.root=root
            controller.settings=dict(reference_root=str(reference),pilot_seeds=[0,1,2],
                confirmation_seeds=[10,11,12,13,14],minimum_improvement=.01,primary_alpha=.025,secondary_alpha=.025)
            controller.jobs={};controller.failed=[];controller.decisions={}
            strong=dict(estimate=.02,seed_points=[.02,.03,.01],pvalue=.001,low=.005,high=.035)
            weak=dict(estimate=.001,seed_points=[.001,.001,.001],pvalue=.5,low=-.01,high=.01)
            def comparisons(phase,study,seeds):
                if study=='secondary':return {k:dict(weak) for k in SECONDARY}
                result={k:dict(strong) for k in PRIMARY}
                if phase=='confirmation':
                    result['content']['low']=-.001
                    result['branch']['pvalue']=.5
                return result
            with patch.object(controller,'stage',return_value=True),patch.object(controller,'family') as family,\
                 patch.object(controller,'comparisons',side_effect=comparisons):
                self.assertEqual(controller.run(),0)
            self.assertEqual(family.call_count,3)
            self.assertEqual(family.call_args.args,('confirmation','primary',[10,11,12,13,14]))
            supported=controller.decisions['primary']['confirmation']['supported']
            self.assertTrue(supported['pairing']);self.assertTrue(supported['overall'])
            self.assertFalse(supported['content']);self.assertFalse(supported['branch'])
            self.assertFalse(controller.decisions['secondary']['advance'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires isolated model environment')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from iglm_baseline import MODEL_CONFIG
        torch.set_num_threads(1)
        cls.config=dict(MODEL_CONFIG,n_layer=1,n_head=4,n_embd=32,n_inner=64,n_positions=128,n_ctx=128)

    def test_shuffle_preserves_antigen_marginals_separately_for_each_selector(self):
        from collections import Counter
        from types import SimpleNamespace
        from controlled_antigen_worker import Scoring
        rows=[record(str(i)) for i in range(6)]
        for i,r in enumerate(rows):r['example_target_chains']='heavy;light' if i<4 else 'heavy'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_json(root/'protocol.json',dict(settings=dict(memory_tokens=32)))
            scorer=Scoring(root,SimpleNamespace(branch=True),dict(memory='shuffled',seed=0),'cpu',rows)
            for chain in ('heavy','light'):
                selected=[r for r in rows if chain in r['example_target_chains']]
                donors=[scorer.donors[r['id'],chain] for r in selected]
                self.assertEqual(Counter(r['resolved_ag_seq'] for r in selected),Counter(r['resolved_ag_seq'] for r in donors))
                self.assertTrue(all(r['resolved_ag_seq']!=d['resolved_ag_seq'] for r,d in zip(selected,donors)))

    def test_exact_shared_weights_and_zero_output_equality_for_both_selectors(self):
        import torch
        from controlled_antigen_model import build,memory,mode
        from antigen_diagnostics import iglm_token_nll
        base=build(4,config=self.config);branch=build(4,branch=True,config=self.config)
        for name,value in base.lm.state_dict().items():self.assertTrue(torch.equal(value,branch.lm.state_dict()[name]),name)
        ag=memory(torch.randn(17,1280));mode(base,False);mode(branch,False)
        self.assertEqual(ag.shape,(1,32,1280))
        for chain in ('heavy','light'):
            expected=iglm_token_nll(base,chain,'ACDEFGHIKLMN')
            actual=iglm_token_nll(branch,chain,'ACDEFGHIKLMN',ag)
            self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(torch.count_nonzero(memory(torch.randn(71,1280),constant=True)),0)

    def test_frozen_decoder_has_active_dropout_and_stays_fixed_during_learning(self):
        import torch
        from controlled_antigen_model import build,memory,mode,parameters,parity,save,load
        from antigen_diagnostics import iglm_token_nll
        model=build(3,branch=True,frozen=True,config=self.config);mode(model,True)
        self.assertTrue(all(m.training for m in model.lm.modules() if isinstance(m,torch.nn.Dropout)))
        original={n:p.detach().clone() for n,p in model.lm.named_parameters() if not p.requires_grad}
        ag=memory(torch.randn(43,1280));optimizer=torch.optim.Adam([p for p in parameters(model) if p.requires_grad],lr=.01)
        for _ in range(2):
            optimizer.zero_grad();iglm_token_nll(model,'heavy','ACDEFGHIKLMNPQRSTVWY',ag).mean().backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,dict(model.lm.named_parameters())[n]) for n,v in original.items()))
        self.assertGreater(float(model.lm.transformer.h[0].crossattention.c_proj.weight.detach().abs().sum()),0)
        mode(model,False)
        self.assertFalse(torch.equal(iglm_token_nll(model,'heavy','ACDEFGHIKLMN',ag),
                                     iglm_token_nll(model,'heavy','ACDEFGHIKLMN',ag+1)))
        for chain in ('heavy','light'):
            result=parity(model,ag,['ACDEFGHIKLMNPQRSTVWY','ACDEFGHIKLMNPQ'],chain)
            self.assertLess(result['max_nll_error'],3e-6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'best.pt';spec=dict(seed=3,branch=True,frozen=True)
            save(path,model,spec,{})
            restored,_=load(path)
            self.assertFalse(restored.lm.training)
            self.assertTrue(torch.equal(iglm_token_nll(model,'light','ACDEFGHIK',ag),
                                        iglm_token_nll(restored,'light','ACDEFGHIK',ag)))

    def test_future_residues_cannot_change_earlier_likelihoods(self):
        import torch
        from controlled_antigen_model import build,memory,mode
        from antigen_diagnostics import iglm_token_nll
        model=build(2,branch=True,config=self.config)
        with torch.no_grad():model.lm.transformer.h[0].crossattention.c_proj.weight.fill_(.01)
        mode(model,False);ag=memory(torch.randn(13,1280))
        a=iglm_token_nll(model,'heavy','ACDEFGHIKLMNPQ',ag)
        b=iglm_token_nll(model,'heavy','ACDEFGHIKLAAAA',ag)
        self.assertTrue(torch.equal(a[:10],b[:10]))


if __name__=='__main__':unittest.main()
