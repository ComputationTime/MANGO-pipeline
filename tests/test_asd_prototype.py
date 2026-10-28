"""HER2 exclusion, deterministic antigen diversity and post-sampling targets."""
import copy
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from prepare_asd_prototype import diverse_subset, her2_antigens
from prepare_asd_transfer import select_examples


class PrototypeTests(unittest.TestCase):
    def test_excludes_her2_across_sources_including_isoforms_and_buzz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('CREATE TABLE rows(row_id INTEGER,antigen_ids TEXT,source TEXT);'
                                 'CREATE TABLE targets(row_id INTEGER,target_name TEXT);')
                cases = [('human HER2', 's1', 'a'), ('ERBB2_HUMAN', 's2', 'b'),
                         ('erb-b2 receptor', 's3', 'c'), ('HER-2|VEGFA', 's4', 'd'),
                         ('EGFR', 's5', 'e'), ('', 'buzz', 'f'), ('HER3', 's6', 'g')]
                for i, (name, source, ag) in enumerate(cases):
                    db.execute('INSERT INTO rows VALUES (?,?,?)', (i, ag, source))
                    db.execute('INSERT INTO targets VALUES (?,?)', (i, name))
            self.assertEqual(her2_antigens(path), {'a', 'b', 'c', 'd', 'f'})

    def test_antigen_round_robin_limits_dominant_target_and_preserves_splits(self):
        rows = [dict(id=f'a{i}', antigen_id='a', split='train', ab_ag_cluster='group1') for i in range(100)]
        rows += [dict(id=f'{ag}{i}', antigen_id=ag, split='train', ab_ag_cluster='group2')
                 for ag in ('b', 'c', 'd') for i in range(3)]
        selected = diverse_subset(rows, 8, per_antigen=2, seed=2)
        self.assertEqual(len(selected), 8)
        self.assertEqual({r['antigen_id'] for r in selected[:4]}, {'a', 'b', 'c', 'd'})
        self.assertLessEqual(sum(r['antigen_id'] == 'a' for r in selected), 2)
        self.assertEqual(selected, diverse_subset(list(reversed(rows)), 8, per_antigen=2, seed=2))
        self.assertTrue(all(r['split'] == 'train' for r in selected))
        self.assertEqual({r['ab_ag_cluster'] for r in selected}, {'group1', 'group2'})

    def test_sampling_recomputes_masks_so_shared_light_target_is_retained(self):
        rows = [dict(id='a', split='train', resolved_ag_seq='ACD', resolved_H_seq='GHI',
                     resolved_L_seq='KLM', example_target_chains='heavy')]
        selected = select_examples(copy.deepcopy(rows))
        self.assertEqual(selected[0]['example_target_chains'], 'heavy;light')
        self.assertEqual(rows[0]['example_target_chains'], 'heavy')


if __name__ == '__main__':
    unittest.main()
