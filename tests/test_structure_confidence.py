import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))
import structure_confidence_common as sc


class StructureConfidenceTests(unittest.TestCase):
    def test_design_selection_is_reproducible_and_order_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.csv"
            second = Path(tmp) / "second.csv"
            rows = pd.DataFrame({
                "embedder": ["one_hot"] * 5,
                "run_id": ["run"] * 5,
                "target_id": ["target"] * 5,
                "design_index": list(range(5)),
                "sequence": ["EVQLV"] * 5,
                "status": ["ok"] * 5,
            })
            rows.to_csv(first, index=False)
            rows.sample(frac=1, random_state=4).to_csv(second, index=False)
            a = sc.select_designs(first, 3, 13)
            b = sc.select_designs(second, 3, 13)
            self.assertEqual(list(a.design_index), list(b.design_index))
            self.assertEqual(len(a), 3)

    def test_target_context_uses_light_and_all_antigen_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records.csv"
            pd.DataFrame([{
                "id": "target", "resolved_L_seq": "DIQMT",
                "resolved_ag_seq": "ACDE,FGHI",
            }]).to_csv(records, index=False)
            antigen, light = sc.target_context(records, "target")
            self.assertEqual(light, "DIQMT")
            self.assertEqual(antigen, ["ACDE", "FGHI"])
            self.assertEqual(
                sc.complex_chains("EVQLV", light, antigen),
                [("H", "EVQLV"), ("L", "DIQMT"),
                 ("AG1", "ACDE"), ("AG2", "FGHI")],
            )

    def test_noncanonical_generated_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            sc.clean_sequence("EVQLX")


if __name__ == "__main__":
    unittest.main()
