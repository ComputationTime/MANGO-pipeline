import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from generation_targets import select_generation_targets


class GenerationTargetTests(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame({
            "id": ["train-a", "val-b", "test-c2", "test-c1", "test-d"],
            "split": ["train", "val", "test", "test", "test"],
            "ab_ag_cluster": ["a", "b", "c", "c", "d"],
        })

    def test_one_representative_from_every_held_out_test_cluster(self):
        selected = select_generation_targets(
            self.records, "test", "one_per_cluster", "ab_ag_cluster")
        self.assertEqual(selected, ["test-c1", "test-d"])
        self.assertFalse(set(selected) & {"train-a", "val-b"})

    def test_cap_is_applied_after_cluster_diversification(self):
        selected = select_generation_targets(
            self.records, "test", "one_per_cluster", "ab_ag_cluster", 1)
        self.assertEqual(selected, ["test-c1"])

    def test_test_cluster_seen_in_training_is_rejected(self):
        broken = pd.concat([
            self.records,
            pd.DataFrame({"id": ["train-c"], "split": ["train"],
                          "ab_ag_cluster": ["c"]}),
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "cluster leakage"):
            select_generation_targets(
                broken, "test", "one_per_cluster", "ab_ag_cluster")


if __name__ == "__main__":
    unittest.main()
