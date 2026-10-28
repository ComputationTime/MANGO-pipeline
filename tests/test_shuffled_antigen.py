import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from shuffle_antigen_embeddings import build_shuffled_embeddings, derangement


class ShuffledAntigenTests(unittest.TestCase):
    def test_derangement_is_deterministic_and_has_no_fixed_points(self):
        ids = [f"record-{index}" for index in range(12)]
        first = derangement(ids, 7, "train")
        second = derangement(ids, 7, "train")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(ids))
        self.assertTrue(all(target != donor for target, donor in zip(ids, first)))

    def test_builder_uses_different_donor_within_each_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.csv"
            source = root / "source"
            output = root / "output"
            rows = []
            for split in ("train", "val", "test"):
                for index in range(3):
                    record_id = f"{split}-{index}"
                    rows.append({"id": record_id, "split": split})
                    path = source / split / f"{record_id}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "embedding": torch.full((index + 1, 4), float(index)),
                        "shape": [index + 1, 4],
                        "id": record_id,
                        "embedder": "esm2",
                        "model_name": "esm2_t33_650M_UR50D",
                    }, path)
            with records.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "split"])
                writer.writeheader()
                writer.writerows(rows)
            source_eval = root / "esm2_eval.json"
            source_eval.write_text(json.dumps({"embedder": "esm2"}))
            marker = output / ".batch_complete.json"
            mapping = output / "shuffle_map.csv"
            build_shuffled_embeddings(
                records, source, source_eval, output, marker, mapping,
                "control_shuffled", "esm2", 0, ["train", "val", "test"],
            )
            with mapping.open(newline="") as handle:
                mapped = list(csv.DictReader(handle))
            self.assertEqual(len(mapped), 9)
            self.assertTrue(all(row["target_equals_donor"] == "False" for row in mapped))
            self.assertTrue(all(row["target_id"].split("-")[0] == row["donor_id"].split("-")[0] for row in mapped))
            payload = torch.load(
                output / "train" / "train-0.pt",
                map_location="cpu", weights_only=False,
            )
            self.assertEqual(payload["embedder"], "control_shuffled")
            self.assertFalse(payload["target_antigen_used"])
            self.assertFalse(payload["pairing_correct"])


if __name__ == "__main__":
    unittest.main()
