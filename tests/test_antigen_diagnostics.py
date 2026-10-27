"""Diagnostic perturbations must preserve targets, positions and split safety."""
from collections import Counter
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from antigen_diagnostics import balanced_records, mango_token_nll, wrong_antigen_map
from model_common import MangoModel
from iglm_baseline import IgLMScratch, MODEL_CONFIG
from train_antigen_diagnostic import build_model, parameter_list
from antigen_diagnostics import iglm_token_nll
from summarize_antigen_diagnostics import paired_difference
from number_diagnostic_regions import region


class DiagnosticTests(unittest.TestCase):
    def test_donors_stay_within_split_and_change_antigen_family(self):
        rows = [dict(id=f'{split}_{i}', split=split, resolved_ag_seq='A' * (i + 2),
                     ab_ag_cluster=f'{split}_{i}')
                for split in ('train', 'val', 'test') for i in range(5)]
        mapping = wrong_antigen_map(rows, 0)
        self.assertEqual(mapping, wrong_antigen_map(rows, 0))
        self.assertNotEqual(mapping, wrong_antigen_map(rows, 1))
        for row in rows:
            donor = mapping[row['id']]
            self.assertEqual(donor['split'], row['split'])
            self.assertNotEqual(donor['resolved_ag_seq'], row['resolved_ag_seq'])
            self.assertNotEqual(donor['ab_ag_cluster'], row['ab_ag_cluster'])

    def test_inspection_subsample_covers_groups_and_keeps_original_records(self):
        rows = [dict(id=f'{cluster}_{i}', ab_ag_cluster=cluster, split='train')
                for cluster in ('a', 'b', 'c') for i in range(10)]
        selected = balanced_records(rows, 6, 0)
        self.assertEqual(Counter(r['ab_ag_cluster'] for r in selected), dict(a=2, b=2, c=2))
        self.assertEqual(len({r['id'] for r in selected}), 6)
        self.assertTrue(all(r in rows for r in selected))

    def test_per_token_scorer_matches_production_loss_for_both_architectures(self):
        torch.set_num_threads(2)
        for architecture in (MangoModel.ARCHITECTURE, MangoModel.CROSS_ATTENTION_ARCHITECTURE):
            with self.subTest(architecture=architecture):
                model = MangoModel(8, architecture=architecture).eval()
                antigen = torch.randn(1, 7, 8)
                with torch.no_grad():
                    scores = mango_token_nll(model, antigen, 'heavy', 'ACD')
                    self.assertEqual(scores.shape, (4,))
                    expected = model.loss(antigen, 'heavy', model.target_token_ids('ACD'))
                    torch.testing.assert_close(scores.mean(), expected)
                    masked = mango_token_nll(model, antigen, 'heavy', 'ACD', standard_causal_mask=True)
                    torch.testing.assert_close(scores, masked, rtol=0, atol=1e-6)
                    empty = mango_token_nll(model, antigen, 'heavy', 'ACD', 'no_context')
                    self.assertEqual(empty.shape, (4,))
                    self.assertTrue(torch.isfinite(empty).all())

    def test_zero_cross_attention_preserves_iglm_and_frozen_training_keeps_base(self):
        torch.set_num_threads(2)
        base = IgLMScratch(dict(initialization='random_from_scratch',
                              objective='full_chain_autoregressive_v1', species_token='[HUMAN]'))
        base.lm.eval()
        source = dict(metadata=dict(model_config=MODEL_CONFIG, public_pretrained_weights_loaded=False),
                      lm=base.lm.state_dict())
        spec = dict(kind='iglm_cross', freeze_decoder=True)
        adapted = build_model(spec, source)
        adapted.lm.eval()
        antigen = torch.randn(1, 9, 1280)
        with torch.no_grad():
            expected = iglm_token_nll(base, 'heavy', 'ACDE')
            torch.testing.assert_close(iglm_token_nll(adapted, 'heavy', 'ACDE', antigen), expected, rtol=0, atol=1e-6)
            torch.testing.assert_close(iglm_token_nll(adapted, 'heavy', 'ACDE', -antigen), expected, rtol=0, atol=1e-6)
        optimizer = torch.optim.Adam([p for p in parameter_list(adapted, 'iglm_cross') if p.requires_grad], lr=1e-3)
        adapted.lm.train()
        iglm_token_nll(adapted, 'heavy', 'ACDE', antigen).mean().backward()
        optimizer.step()
        for key, value in base.lm.state_dict().items():
            torch.testing.assert_close(adapted.lm.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(any(torch.count_nonzero(block.crossattention.c_proj.weight) for block in adapted.lm.transformer.h))

    def test_rebased_positions_retain_antigen_attention_and_causal_target_mask(self):
        torch.set_num_threads(2)
        model = MangoModel(8).eval()
        antigen = torch.randn(1, 7, 8, requires_grad=True)
        scores = mango_token_nll(model, antigen, 'heavy', 'ACD', rebase=True)
        scores.sum().backward()
        self.assertGreater(float(antigen.grad.abs().sum()), 1e-5)
        with torch.no_grad():
            swapped = mango_token_nll(model, -antigen, 'heavy', 'ACD', rebase=True)
            self.assertGreater(float((swapped - scores.detach()).abs().max()), 1e-5)
            changed_tail = mango_token_nll(model, antigen, 'heavy', 'ACE', rebase=True)
            # Last target replacement must not alter predictions of A and C.
            torch.testing.assert_close(changed_tail[:2], scores.detach()[:2], rtol=0, atol=1e-6)

    def test_cluster_bootstrap_pairs_exact_targets_and_preserves_token_weighting(self):
        left = [dict(id='a', chain='heavy', cluster='one', all_sum=2., all_tokens=2),
                dict(id='b', chain='light', cluster='two', all_sum=12., all_tokens=4)]
        right = [dict(id='a', chain='heavy', cluster='one', all_sum=1., all_tokens=2),
                 dict(id='b', chain='light', cluster='two', all_sum=8., all_tokens=4)]
        result = paired_difference(left, right, 'all', 500, 0)
        self.assertAlmostEqual(result['delta_nll'], 5 / 6)
        self.assertEqual(result['n_clusters'], 2)
        self.assertEqual(result['cluster_bootstrap_95ci'], [.5, 1.])
        with self.assertRaises(ValueError):
            paired_difference(left, right[:1], 'all', 500, 0)
        with self.assertRaises(ValueError):
            paired_difference(left, [dict(right[0], all_tokens=3), right[1]], 'all', 500, 0)

    def test_imgt_region_boundaries_are_inclusive_and_eos_is_separate(self):
        self.assertEqual([region(p) for p in (26, 27, 38, 39, 55, 56, 65, 66, 104, 105, 117, 118)],
                         ['framework', 'cdr1', 'cdr1', 'framework', 'framework', 'cdr2', 'cdr2',
                          'framework', 'framework', 'cdr3', 'cdr3', 'framework'])


if __name__ == '__main__':
    unittest.main()
