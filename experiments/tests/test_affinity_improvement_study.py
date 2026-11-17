"""Predeclared decision, objective, data matching and unattended-resume contracts."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from affinity_study_common import (advancement,freeze,holm,paired_group_bootstrap,
    read_json,retry_kind,training_weights,write_json)
from prepare_affinity_study import affinity_audit, diversity_match,length_signature


class DecisionTests(unittest.TestCase):
    settings=dict(pilot_seeds=[0,1,2],minimum_development_groups=20,affinity_improvement=.02,nll_improvement=.01)

    def test_missing_seeds_and_small_biological_samples_do_not_advance(self):
        self.assertFalse(advancement([.1,.1],self.settings,100)['advance'])
        self.assertFalse(advancement([.1,.1,.1],self.settings,19)['advance'])
        self.assertFalse(advancement([.1,-.01,-.01],self.settings,100)['advance'])
        self.assertTrue(advancement([.015,.015,.015],self.settings,100)['advance'])
        self.assertFalse(advancement([.015,.015,.015],self.settings,100,True)['advance'])

    def test_holm_keeps_infeasible_hypotheses_in_the_family(self):
        result=holm(dict(balance=.009,diversity=.02,decoder=.04,ranking=1.,cdr=1.))
        self.assertEqual(result,dict(balance=True,diversity=False,decoder=False,ranking=False,cdr=False))

    def test_bootstrap_pairs_groups_and_preserves_token_weighting(self):
        rows=[dict(improvement_sum=1.,ntokens=100),dict(improvement_sum=9.,ntokens=300)]
        result=paired_group_bootstrap(rows,200,seed=0)
        self.assertAlmostEqual(result['improvement'],.025)
        self.assertGreater(result['low'],0)
        self.assertEqual(result['groups'],2)
        self.assertEqual(paired_group_bootstrap(rows[:1],20)['pvalue'],1.)

    def test_only_transient_failures_retry(self):
        self.assertEqual(retry_kind('Connection reset by peer'),'transient')
        self.assertEqual(retry_kind('CUDA out of memory'),'oom')
        self.assertIsNone(retry_kind('ValueError: invalid sequence'))

    def test_manifest_is_immutable_and_resume_does_not_rewrite(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'protocol.json';freeze(path,dict(seed=0));mtime=path.stat().st_mtime_ns
            freeze(path,dict(seed=0));self.assertEqual(path.stat().st_mtime_ns,mtime)
            with self.assertRaises(ValueError):freeze(path,dict(seed=1))


class DataTests(unittest.TestCase):
    def test_balancing_retains_membership_and_normalizes_gradient_scale(self):
        examples=[dict(antigen_id='a')]*3+[dict(antigen_id='b')]
        weights=training_weights(examples,True)
        self.assertAlmostEqual(sum(weights),4.)
        self.assertAlmostEqual(sum(weights[:3]),weights[3])

    def test_diversity_matching_keeps_exact_chain_token_budget(self):
        def row(name,ag,h=3,l=2):
            return dict(id=name,resolved_ag_seq=ag,resolved_H_seq='A'*h,resolved_L_seq='C'*l,
                        example_target_chains='heavy;light')
        base=[row('one','AG'),row('two','AG')]
        pool=base+[row('three','AA'),row('four','AC',4)]
        selected=diversity_match(base,pool)
        self.assertEqual(Counter(map(length_signature,base)),Counter(map(length_signature,selected)))
        self.assertGreater(len({r['resolved_ag_seq'] for r in selected}),1)

    def test_missing_assay_provenance_cannot_become_ranking_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            result=affinity_audit(dict(verified_affinity_records=None),[],Path(directory))
            self.assertFalse(result['feasible'])
            path=Path(directory)/'unverified.csv';path.write_text('id,log10_kd_nm\na,1\n')
            with self.assertRaises(ValueError):
                affinity_audit(dict(verified_affinity_records=str(path)),[],Path(directory))


@unittest.skipUnless(importlib.util.find_spec('torch'),'run gradient tests in model environment')
class ObjectiveTests(unittest.TestCase):
    def test_ranking_gradient_favors_stronger_binder(self):
        import torch
        from affinity_study_worker import ranking_loss
        strong=torch.tensor(2.,requires_grad=True);weak=torch.tensor(2.,requires_grad=True)
        ranking_loss(strong,weak).backward()
        self.assertGreater(strong.grad.item(),0)
        self.assertLess(weak.grad.item(),0)

    def test_cdr_weighting_keeps_framework_and_eos(self):
        import torch
        from affinity_study_worker import region_objective
        losses=torch.tensor([1.,2.,3.],requires_grad=True)
        loss=region_objective(losses,['framework','cdr3','eos'],2.)
        self.assertAlmostEqual(loss.item(),2.)
        loss.backward();self.assertEqual(losses.grad.tolist(),[.25,.5,.25])
        with self.assertRaises(ValueError):region_objective(losses,['cdr3'],2.)


if __name__=='__main__':unittest.main()
