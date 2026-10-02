import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from embed_antigen_control import antigen_control, control_matrix


class AntigenControlTests(unittest.TestCase):
    def test_constant_is_identical_and_contains_no_record_information(self):
        spec = {"mode": "constant", "width": 7, "tokens": 1, "seed": 0}
        first = control_matrix("record-a", spec)
        second = control_matrix("record-b", spec)
        self.assertEqual(tuple(first.shape), (1, 7))
        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(first, second))

    def test_random_is_deterministic_per_record_and_differs_between_records(self):
        spec = {"mode": "random", "width": 32, "tokens": 1, "seed": 11}
        first = control_matrix("record-a", spec)
        repeated = control_matrix("record-a", spec)
        other = control_matrix("record-b", spec)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other))

    def test_saved_metadata_declares_no_antigen_source(self):
        spec = {"mode": "random", "width": 5, "tokens": 1, "seed": 3}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "embedding.pt"
            antigen_control(
                "unused.csv", "record-a", str(output), "none", "control_random",
                spec, row={"id": "record-a", "resolved_ag_seq": "SHOULDNOTBEUSED"},
            )
            payload = torch.load(output, map_location="cpu", weights_only=False)
        self.assertFalse(payload["source_antigen_used"])
        self.assertEqual(payload["conditioning_scope"], "no_antigen_fixed_token_v1")
        self.assertEqual(payload["chains"], [])
        self.assertEqual(payload["shape"], [1, 5])


if __name__ == "__main__":
    unittest.main()
