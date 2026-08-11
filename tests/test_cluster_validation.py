import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from process_records import (
    assign_validation,
    select_cluster_subset,
    validate_cluster_partition,
)


class ClusterValidationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "c1", "c2", "test1"],
                "split": ["train", "train", "train", "train", "train", "test"],
                "ab_ag_cluster": ["a", "a", "b", "c", "c", "d"],
            }
        )
        self.cfg = {
            "strategy": "cluster",
            "fraction": 1 / 3,
            "seed": 13,
            "cluster_column": "ab_ag_cluster",
        }

    def test_clusters_do_not_cross_train_and_validation(self):
        split = assign_validation(self.df, self.cfg)
        train_clusters = set(self.df.loc[split == "train", "ab_ag_cluster"])
        val_clusters = set(self.df.loc[split == "val", "ab_ag_cluster"])
        self.assertTrue(val_clusters)
        self.assertFalse(train_clusters & val_clusters)
        self.assertEqual(split.iloc[-1], "test")

    def test_assignment_is_independent_of_row_order(self):
        original = assign_validation(self.df, self.cfg)
        expected = dict(zip(self.df["id"], original))
        shuffled = self.df.sample(frac=1, random_state=99).reset_index(drop=True)
        actual = assign_validation(shuffled, self.cfg)
        self.assertEqual(expected, dict(zip(shuffled["id"], actual)))

    def test_missing_cluster_column_fails(self):
        with self.assertRaisesRegex(ValueError, "requires column"):
            assign_validation(self.df.drop(columns=["ab_ag_cluster"]), self.cfg)

    def test_blank_training_cluster_fails(self):
        broken = self.df.copy()
        broken.loc[0, "ab_ag_cluster"] = ""
        with self.assertRaisesRegex(ValueError, "without 'ab_ag_cluster'"):
            assign_validation(broken, self.cfg)

    def test_source_test_cluster_cannot_overlap_training(self):
        broken = self.df.copy()
        broken.loc[broken["split"] == "test", "ab_ag_cluster"] = "a"
        split = assign_validation(broken, self.cfg)
        with self.assertRaisesRegex(ValueError, "leakage across train/val/test"):
            validate_cluster_partition(broken, split, self.cfg)

    def test_smoke_subset_selects_shortest_rows_from_distinct_clusters(self):
        frame = pd.DataFrame(
            {
                "id": ["a-long", "a-short", "b", "c", "d", "e", "x", "y"],
                "split": ["train"] * 6 + ["test"] * 2,
                "ab_ag_cluster": ["a", "a", "b", "c", "d", "e", "x", "y"],
                "resolved_ag_seq": ["A" * 20, "A" * 5, "A" * 6, "A" * 7,
                                    "A" * 8, "A" * 9, "A" * 5, "A" * 6],
            }
        )
        subset = select_cluster_subset(
            frame,
            {"enabled": True, "train_clusters": 4, "test_clusters": 1,
             "rows_per_cluster": 1},
            self.cfg,
        )
        self.assertEqual(set(subset.id), {"a-short", "b", "c", "d", "x"})
        self.assertEqual(len(subset), 5)


if __name__ == "__main__":
    unittest.main()
