import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pool_antigen_embeddings import build_pooled_embeddings


class PooledAntigenTests(unittest.TestCase):
    def test_builder_mean_pools_correct_record_to_one_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.csv"
            source = root / "source"
            output = root / "output"
            rows = []
            for split in ("train", "val", "test"):
                record_id = f"{split}-record"
                rows.append({"id": record_id, "split": split})
                path = source / split / f"{record_id}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "embedding": torch.tensor([[1.0, 3.0], [3.0, 7.0]]),
                    "shape": [2, 2],
                    "id": record_id,
                    "embedder": "esm2",
                    "model_name": "esm2_t33_650M_UR50D",
                    "chains": ["A"],
                }, path)
            with records.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "split"])
                writer.writeheader()
                writer.writerows(rows)
            source_eval = root / "esm2_eval.json"
            source_eval.write_text(json.dumps({"embedder": "esm2"}))
            marker = output / ".batch_complete.json"
            build_pooled_embeddings(
                records, source, source_eval, output, marker,
                "control_pooled_esm2", "esm2", ["train", "val", "test"],
            )
            payload = torch.load(
                output / "train" / "train-record.pt",
                map_location="cpu", weights_only=False,
            )
            torch.testing.assert_close(
                payload["embedding"], torch.tensor([[2.0, 5.0]])
            )
            self.assertEqual(payload["shape"], [1, 2])
            self.assertEqual(payload["source_length"], 2)
            self.assertTrue(payload["target_antigen_used"])
            self.assertEqual(payload["pooling"], "mean_over_antigen_tokens_v1")


if __name__ == "__main__":
    unittest.main()
