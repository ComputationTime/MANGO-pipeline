"""Dominant-antigen controls must retain exact marginals and disclose fixed pairs."""
from collections import Counter
import json
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from asd_scaling_studies import maximal_record_shuffle, studies_in_order
from prepare_asd_scaling import record_derangement
from prepare_asd_transfer import sha256
from report_asd_scaling_8k import verify_training_shuffle


class ASD16kTests(unittest.TestCase):
    def test_maximum_shuffle_attains_bound_and_preserves_splits_and_marginals(self):
        for n in range(3, 20):
            for majority in range((n + 1) // 2, n):
                with self.subTest(n=n, majority=majority):
                    rows = [dict(id=f'{split}_{i}', split=split,
                                 resolved_ag_seq='A' if i < majority else f'S{i}')
                            for split in ('train', 'val') for i in range(n)]
                    donors = maximal_record_shuffle(rows)
                    self.assertEqual(donors, maximal_record_shuffle(list(reversed(rows))))
                    for split in ('train', 'val'):
                        members = [r for r in rows if r['split'] == split]
                        self.assertEqual(Counter(r['id'] for r in members), Counter(donors[r['id']]['id'] for r in members))
                        self.assertTrue(all(donors[r['id']]['split'] == split for r in members))
                        self.assertEqual(sum(r['resolved_ag_seq'] == donors[r['id']]['resolved_ag_seq'] for r in members),
                                         max(0, 2 * majority - n))

    def test_balanced_shuffle_matches_the_frozen_control_exactly(self):
        rows = [dict(id=str(i), split='train', resolved_ag_seq='ABC'[i // 4]) for i in range(12)]
        for seed in (0, 1):
            self.assertEqual(maximal_record_shuffle(rows, seed), record_derangement(rows, seed))

    def test_no_antigen_identity_change_possible_is_rejected(self):
        rows = [dict(id=str(i), split='train', resolved_ag_seq='A') for i in range(4)]
        with self.assertRaises(ValueError):
            maximal_record_shuffle(rows)

    def test_report_checks_the_actual_map_against_the_claimed_fixed_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / 'n6/cohort'
            cohort.mkdir(parents=True)
            rows = [dict(id=str(i), split='train', resolved_ag_seq='A' if i < 4 else 'B') for i in range(6)]
            rows += [dict(id='heldout', split='val', resolved_ag_seq='C')]
            with (cohort / 'asd_records.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            donors = maximal_record_shuffle(rows[:6])
            path = root / 'n6/shuffled/antigen/esm2_shuffled/mapping.csv'
            path.parent.mkdir(parents=True)
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['id', 'donor_id']); writer.writeheader()
                writer.writerows(dict(id=r['id'], donor_id=donors[r['id']]['id'] if r['split']=='train' else r['id']) for r in rows)
            marker = dict(shuffled_mapping_sha256=sha256(path), shuffled_correct_antigen_records=2)
            marker_path = root / 'n6/embeddings_complete.json'
            marker_path.write_text(json.dumps(marker))
            study = dict(experiment_root=str(root), allow_partial_training_shuffle=True)
            self.assertEqual(verify_training_shuffle(study, 6)['correct_antigen_records'], 2)
            marker['shuffled_correct_antigen_records'] = 0
            marker_path.write_text(json.dumps(marker))
            with self.assertRaises(ValueError):
                verify_training_shuffle(study, 6)

    def test_reference_chain_resolves_in_order_and_rejects_cycles_and_run_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base.json'
            base = dict(experiment_root='base', runs={'old': {}})
            path.write_text(json.dumps(base))
            current = dict(experiment_root='new', runs={'new': {}}, reference_config=str(path))
            self.assertEqual(studies_in_order(current), [base, current])
            with self.assertRaises(ValueError):
                studies_in_order(dict(current, runs={'old': {}}))
            path.write_text(json.dumps(dict(base, reference_config=str(path))))
            with self.assertRaises(ValueError):
                studies_in_order(current)


if __name__ == '__main__':
    unittest.main()
