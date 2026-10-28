"""Scaling perturbations preserve split and antigen marginal contracts."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from prepare_asd_scaling import record_derangement
from prepare_asd_prototype import diverse_subset
from prepare_asd_transfer import select_examples
from report_asd_scaling import average_rows, difference_rows
from summarize_antigen_diagnostics import REGIONS, paired_difference


class ASDScalingTests(unittest.TestCase):
    def test_donors_preserve_marginals_change_antigen_and_never_cross_splits(self):
        rows = [dict(id=f'{split}_{i}', split=split, resolved_ag_seq='A' * (2 + i // 3))
                for split in ('train', 'val', 'test') for i in range(12)]
        mapping = record_derangement(rows)
        self.assertEqual(mapping, record_derangement(list(reversed(rows))))
        self.assertNotEqual(mapping, record_derangement(rows, 1))
        for row in rows:
            self.assertEqual(row['split'], mapping[row['id']]['split'])
            self.assertNotEqual(row['resolved_ag_seq'], mapping[row['id']]['resolved_ag_seq'])
        for split in ('train', 'val', 'test'):
            members = [r for r in rows if r['split'] == split]
            self.assertEqual(Counter(r['id'] for r in members), Counter(mapping[r['id']]['id'] for r in members))
            self.assertEqual(Counter(r['resolved_ag_seq'] for r in members),
                             Counter(mapping[r['id']]['resolved_ag_seq'] for r in members))

    def test_impossible_derangement_rejected(self):
        rows = [dict(id=str(i), split='train', resolved_ag_seq='A' if i < 4 else 'B') for i in range(5)]
        with self.assertRaises(ValueError):
            record_derangement(rows)

    def test_dedup_before_size_prefix_keeps_targets_nested(self):
        rows = [dict(id=f'{ag}_{i}', antigen_id=ag, resolved_ag_seq=ag,
                     resolved_H_seq=f'H{i}', resolved_L_seq='shared', split='train')
                for ag in ('AAA', 'CCC', 'DDD') for i in range(8)]
        stream = select_examples(diverse_subset(rows, len(rows), 4, 0))
        small, large = stream[:5], stream[:10]
        self.assertEqual(large[:len(small)], small)
        self.assertLessEqual(max(Counter(r['antigen_id'] for r in large).values()), 4)
        self.assertTrue(all(r['example_target_chains'] in {'heavy', 'heavy;light'} for r in large))

    def test_scale_contrasts_measure_change_in_conditional_gain(self):
        def row(value):
            result = dict(id='target', chain='heavy', cluster='group', variant='correct')
            for region in REGIONS:
                result[region + '_sum'] = value
                result[region + '_tokens'] = 10
            return result
        # Correct context improves by 0.1 nats/token at 2k, and 0.3 at 4k.
        benefit_2k = difference_rows([row(11.)], [row(10.)])
        benefit_4k = difference_rows([row(10.)], [row(7.)])
        change = paired_difference(benefit_4k, benefit_2k, 'all', 20, 0)
        self.assertAlmostEqual(change['delta_nll'], .2)
        averaged = average_rows([row(9.)], [row(11.)], 'wrong_mean')
        self.assertEqual(averaged[0]['all_sum'], 10.)
        with self.assertRaises(ValueError):
            average_rows([row(9.)], [dict(row(11.), all_tokens=11)], 'wrong_mean')


if __name__ == '__main__':
    unittest.main()
