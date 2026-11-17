"""Paired mechanisms, immutable targets, and honest uncertainty/data gates."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from native_mechanism_common import arms,paired_arrays,summarize
from prepare_native_pilot import allocate_single_panel,deduplicate
from affinity_study_common import write_json


class ScientificTests(unittest.TestCase):
    def test_biological_interval_does_not_claim_to_cover_seed_variability(self):
        # Every group has exactly the same effect; the two seeds differ greatly.
        losses=np.vstack([np.ones(20),np.ones(20)*3]);counts=np.ones_like(losses)
        result=summarize(losses,counts,1000,0)
        self.assertEqual((result['mean'],result['low'],result['high']),(2.,2.,2.))
        self.assertAlmostEqual(result['seed_sd'],np.sqrt(2))
        self.assertIn('conditional',result['interval_scope'])
        self.assertNotIn('pvalue',result)

    def test_group_bootstrap_recomputes_token_weighted_ratio(self):
        losses=np.array([[1.,99.]*10]);counts=np.array([[1.,9.]*10])
        result=summarize(losses,counts,1000,0)
        self.assertEqual(result['mean'],10.)
        self.assertLess(result['low'],10.);self.assertGreater(result['high'],10.)
        self.assertIsNone(result['seed_sd'])

    def test_small_group_count_does_not_get_confidence_interval(self):
        result=summarize([[1,2],[3,4]],[[1,1],[1,1]],100)
        self.assertIsNone(result['low']);self.assertIsNone(result['high'])

    def test_affinity_ranks_handle_ties_and_constant_scores(self):
        from native_mechanism_common import affinity_correlations
        values=affinity_correlations([1,2,2,4],[4,3,3,1])
        self.assertAlmostEqual(values['spearman'],-1.)
        self.assertIsNone(affinity_correlations([1,1,1],[1,2,3])['spearman'])

    def test_pairing_refuses_different_sequences_with_same_ids_and_lengths(self):
        row=dict(id='a',group='g',chain='heavy',variant='correct',sequence_sha256='one',ntokens='10',loss_sum='20')
        tables={0:{'a':[row],'b':[dict(row,sequence_sha256='two')]}}
        with self.assertRaisesRegex(ValueError,'target'):paired_arrays(tables,{'a':1.,'b':-1.})
        tables[0]['b']=[dict(row,loss_sum='19')]
        losses,counts=paired_arrays(tables,{'a':1.,'b':-1.})
        self.assertEqual(losses.tolist(),[[1.]])
        self.assertEqual(counts.tolist(),[[10]])

    def test_missing_light_chain_never_becomes_a_training_target(self):
        row=dict(id='a',resolved_H_seq='ACD',resolved_L_seq='',resolved_ag_seq='EFG')
        result=deduplicate([row,dict(row,id='b')])
        self.assertEqual(len(result),1);self.assertEqual(result[0]['example_target_chains'],'heavy')

    def test_single_panel_allocation_keeps_every_old_holdout_reserved(self):
        groups=[]
        for i in range(12):
            groups.append([dict(id=str(i),source_split='reserved' if i<2 else 'train',
                full_biological_group=f'g{i}',ab_ag_cluster=f'g{i}',resolved_ag_seq=f'ag{i}',
                resolved_H_seq=f'h{i}',resolved_L_seq='')])
        cfg=dict(minimum_groups=4,minimum_heavy_examples=4,minimum_training_records=6,minimum_training_antigens=6)
        result=allocate_single_panel(groups,cfg)
        self.assertIsNotNone(result)
        self.assertTrue(all(r['split']=='test' for r in result if r['source_split']=='reserved'))
        self.assertTrue(all(r['source_split']=='train' for r in result if r['split']=='train'))
        self.assertEqual(len({r['biological_group'] for r in result if r['split']=='test'}),4)
        self.assertIsNone(allocate_single_panel(groups,dict(cfg,minimum_training_records=10)))

    def test_infeasible_audit_runs_report_and_never_trains(self):
        from run_native_mechanism import Controller
        with tempfile.TemporaryDirectory() as directory:
            c=Controller.__new__(Controller);c.root=Path(directory);c.cfg={'audit_timeout_seconds':10};c.failed=[];c.jobs={}
            write_json(c.root/'cohort_audit.json',{'feasible':False})
            calls=[]
            with patch.object(c,'stage',side_effect=lambda name,target,*args:calls.append(name) or True):
                self.assertEqual(c.run(),2)
            self.assertEqual(calls,['audit','report'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires isolated model environment')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from iglm_baseline import MODEL_CONFIG
        torch.set_num_threads(1)
        cls.config=dict(MODEL_CONFIG,n_layer=1,n_head=4,n_embd=32,n_inner=64,n_positions=128,n_ctx=128)

    def spec(self,arm):return dict(arms()[arm],seed=100,arm=arm)

    def test_all_branches_initially_preserve_source_likelihood_and_both_selectors(self):
        import torch
        from native_mechanism_worker import build
        from controlled_antigen_model import mode
        from antigen_diagnostics import iglm_token_nll
        native=build(self.spec('source'),config=self.config);mode(native,False)
        source={'weights':{'lm':native.lm.state_dict()},'decoder_config':self.config}
        for arm in ('joint_correct','frozen_correct','onehot_correct'):
            model=build(self.spec(arm),source);mode(model,False)
            memory=torch.randn(1,32,21 if arm.startswith('onehot') else 1280)
            for chain in ('heavy','light'):
                self.assertTrue(torch.allclose(iglm_token_nll(native,chain,'ACDEFGHIK'),
                    iglm_token_nll(model,chain,'ACDEFGHIK',memory),atol=1e-6,rtol=0))

    def test_pairing_controls_match_parameters_and_freezing_preserves_decoder(self):
        import torch
        from native_mechanism_worker import build
        from controlled_antigen_model import mode,parameters
        from antigen_diagnostics import iglm_token_nll
        a=build(self.spec('frozen_correct'),config=self.config);b=build(self.spec('frozen_shuffled'),config=self.config)
        self.assertEqual(sum(p.numel() for p in parameters(a) if p.requires_grad),sum(p.numel() for p in parameters(b) if p.requires_grad))
        snapshot={k:v.clone() for k,v in a.lm.state_dict().items() if '.crossattention.' not in k and '.ln_cross_attn.' not in k}
        old=a.lm.transformer.h[0].crossattention.c_proj.weight.detach().clone();mode(a,True)
        optimizer=torch.optim.Adam([p for p in parameters(a) if p.requires_grad],lr=.001)
        iglm_token_nll(a,'heavy','ACDEFGHIK',torch.randn(1,32,1280)).mean().backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,a.lm.state_dict()[k]) for k,v in snapshot.items()))
        self.assertFalse(torch.equal(old,a.lm.transformer.h[0].crossattention.c_proj.weight))

    def test_widened_unconditional_matches_ESM2_branch_parameter_count(self):
        from native_mechanism_worker import build
        from controlled_antigen_model import parameters
        a=build(self.spec('wide_source'),config=self.config);b=build(self.spec('joint_correct'),config=self.config)
        self.assertLess(abs(sum(p.numel() for p in parameters(a))/sum(p.numel() for p in parameters(b))-1),.001)

    def test_onehot_checkpoint_roundtrip_is_deterministic_and_in_eval_mode(self):
        import torch
        from native_mechanism_worker import build,save,load
        from controlled_antigen_model import mode,parity
        model=build(self.spec('onehot_correct'),config=self.config)
        with torch.no_grad():model.lm.transformer.h[0].crossattention.c_proj.weight.fill_(.01)
        mode(model,False)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'final.pt';save(path,model,self.spec('onehot_correct'),{})
            restored,_=load(path,'cpu');self.assertFalse(restored.lm.training)
            for chain in ('heavy','light'):
                self.assertLess(parity(restored,torch.randn(1,32,21),['ACDEFGHIK','ACDEFGH'],chain)['max_nll_error'],3e-6)


if __name__=='__main__':unittest.main()
