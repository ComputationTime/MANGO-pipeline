import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from embed_batch import batch_embed


class GPUBatchTests(unittest.TestCase):
    def test_batch_writes_contract_and_reuses_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.csv"
            fields = ["id", "split", "antigen_chains", "resolved_ag_seq"]
            with records.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "example",
                        "split": "train",
                        "antigen_chains": "A",
                        "resolved_ag_seq": "ACDE",
                    }
                )
            out_dir = root / "one_hot"
            marker = out_dir / ".batch_complete.json"
            kwargs = dict(
                records_csv=str(records),
                output_dir=str(out_dir),
                marker=str(marker),
                kind="antigen",
                tag="one_hot",
                method="one_hot",
                spec={"method": "one_hot", "class": "naive", "label": "One-hot"},
                seq_source="resolved",
                splits=["train"],
                implementation=str(SCRIPTS / "embed_antigen_one_hot.py"),
            )
            batch_embed(**kwargs)
            first = json.loads(marker.read_text())
            self.assertEqual(first["built"], 1)
            self.assertEqual(first["reused"], 0)
            self.assertTrue((out_dir / "train" / "example.pt").is_file())

            batch_embed(**kwargs)
            second = json.loads(marker.read_text())
            self.assertEqual(second["built"], 0)
            self.assertEqual(second["reused"], 1)


if __name__ == "__main__":
    unittest.main()
