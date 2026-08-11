import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

from standardize_sabdab2 import standardize


class SAbDab2StandardizerTests(unittest.TestCase):
    def test_vh_is_target_and_full_heavy_is_provenance_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "abag_split.csv"
            output = root / "standardized.csv"
            pd.DataFrame(
                [
                    {
                        "INSTANCE": "example_H_L",
                        "agchains": "A",
                        "Hseq_expected": "FULLHEAVYEXPECTED",
                        "Lseq_expected": "LIGHT",
                        "agexpectedseqs": "ANTIGEN",
                        "VH_numerable_seq": "EVQLVESGGG",
                        "Hseq": "FULLHEAVYRESOLVED",
                        "Lseq": "LIGHT",
                        "agresolvedseqs": "ANTIGEN",
                        "ab_ag_split": "train",
                    }
                ]
            ).to_csv(source, index=False)

            result = standardize(
                str(source), str(root), "ab_ag_split", str(output)
            )
            row = result.iloc[0]
            self.assertEqual(row["expected_heavy_seq"], "EVQLVESGGG")
            self.assertEqual(row["resolved_H_seq"], "EVQLVESGGG")
            self.assertEqual(
                row["source_resolved_full_heavy_seq"], "FULLHEAVYRESOLVED"
            )
            self.assertEqual(
                row["source_expected_full_heavy_seq"], "FULLHEAVYEXPECTED"
            )


if __name__ == "__main__":
    unittest.main()
