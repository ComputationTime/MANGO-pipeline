import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from embed_antigen_control import antigen_control, control_matrix, control_model_name
from embed_batch import batch_embed


class AntigenControlTests(unittest.TestCase):
    def source(self, root, record_id="record-a", shape=(4, 7), **overrides):
        path = root / "esm2" / "train" / f"{record_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": record_id,
            "embedder": "esm2",
            "model_name": "esm2_t33_650M_UR50D",
            "embedding": torch.full(shape, float("nan")),
            "shape": list(shape),
        }
        payload.update(overrides)
        torch.save(payload, path)
        return path

    def matched_spec(self, root, mode="constant"):
        return {
            "mode": mode,
            "width": 7,
            "shape_mode": "match_source",
            "source_embedder": "esm2",
            "source_model_name": "esm2_t33_650M_UR50D",
            "source_dir": str(root / "esm2"),
            "seed": 3,
        }

    def embed(self, root, spec, record_id="record-a", **row_fields):
        path = root / "output.pt"
        antigen_control(
            "unused.csv", record_id, str(path), "none", f"control_{spec['mode']}",
            spec, row={"id": record_id, "split": "train", **row_fields},
        )
        return torch.load(path, map_location="cpu", weights_only=False)

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

    def test_both_controls_match_each_source_shape_without_using_source_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for record_id, shape in (("record-a", (4, 7)), ("record-b", (9, 7))):
                self.source(root, record_id, shape)
                for mode in ("constant", "random"):
                    with self.subTest(record_id=record_id, mode=mode):
                        payload = self.embed(root, self.matched_spec(root, mode), record_id)
                        matrix = payload["embedding"]
                        self.assertEqual(tuple(matrix.shape), shape)
                        self.assertEqual(payload["shape"], list(shape))
                        self.assertEqual(matrix.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(matrix).all())
                        self.assertEqual(payload["conditioning_scope"], "no_antigen_values_source_shape_v1")
                        self.assertTrue(payload["source_antigen_shape_used"])
                        self.assertFalse(payload["source_antigen_values_used"])
                        if mode == "constant":
                            self.assertTrue(torch.equal(matrix, torch.zeros(shape)))

    def test_matched_random_depends_on_id_and_seed_but_not_source_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            spec = self.matched_spec(root, "random")
            first = self.embed(root, spec)["embedding"]
            self.source(root, embedding=torch.ones(4, 7))
            second = self.embed(root, spec, resolved_ag_seq="CHANGED", resolved_H_seq="CHANGED")["embedding"]
            self.assertTrue(torch.equal(first, second))
            self.assertFalse(torch.equal(first[0], first[1]))
            self.source(root, "record-b")
            self.assertFalse(torch.equal(first, self.embed(root, spec, "record-b")["embedding"]))
            self.assertFalse(torch.equal(first, self.embed(root, {**spec, "seed": 4})["embedding"]))

    def test_missing_or_incompatible_source_fails_without_single_token_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.matched_spec(root)
            with self.assertRaisesRegex(FileNotFoundError, "shape reference"):
                self.embed(root, spec)
            for overrides, message in (
                ({"id": "wrong"}, "provenance"),
                ({"embedder": "wrong"}, "provenance"),
                ({"model_name": "esm2_t6_8M_UR50D"}, "provenance"),
                ({"embedding": torch.zeros(4, 7), "shape": [1, 7]}, "invalid"),
                ({"embedding": torch.zeros(4, 8), "shape": [4, 8]}, "width"),
            ):
                with self.subTest(overrides=overrides.keys()):
                    self.source(root, **overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        self.embed(root, spec)
                    self.assertFalse((root / "output.pt").exists())

    def test_batch_replaces_legacy_tokens_and_tracks_source_shape_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            records = root / "records.csv"
            with records.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "split"])
                writer.writeheader()
                writer.writerow({"id": "record-a", "split": "train"})
            output_dir = root / "control"
            marker = output_dir / ".batch_complete.json"
            kwargs = dict(
                records_csv=str(records), output_dir=str(output_dir), marker=str(marker),
                kind="antigen", tag="control_constant", method="antigen_control",
                spec={"mode": "constant", "width": 7, "tokens": 1, "seed": 3},
                seq_source="none", splits=["train"],
                implementation=str(SCRIPTS / "embed_antigen_control.py"),
            )
            batch_embed(**kwargs)
            kwargs["spec"] = self.matched_spec(root)
            batch_embed(**kwargs)
            output = output_dir / "train" / "record-a.pt"
            self.assertEqual(torch.load(output, weights_only=False)["shape"], [4, 7])
            self.assertEqual(json.loads(marker.read_text())["built"], 1)
            batch_embed(**kwargs)
            self.assertEqual(json.loads(marker.read_text())["reused"], 1)
            self.source(root, shape=(8, 7))
            newer = output.stat().st_mtime + 1
            os.utime(source, (newer, newer))
            batch_embed(**kwargs)
            self.assertEqual(json.loads(marker.read_text())["built"], 1)
            self.assertEqual(torch.load(output, weights_only=False)["shape"], [8, 7])

    def test_fractional_lengths_round_up_per_record_and_keep_one_token_minimum(self):
        for mode in ("constant", "random"):
            for length in (1, 2, 3, 4, 5, 197, 991):
                for fraction, numerator in ((0.25, 1), (0.5, 2), (0.75, 3), (1, 4)):
                    spec = {"mode": mode, "width": 7, "shape_mode": "match_source", "length_fraction": fraction}
                    matrix = control_matrix("record", spec, shape=(length, 7))
                    with self.subTest(mode=mode, length=length, fraction=fraction):
                        self.assertEqual(tuple(matrix.shape), (max(1, (length * numerator + 3) // 4), 7))

    def test_fractional_noise_is_exact_prefix_of_full_noise_and_has_compact_storage(self):
        spec = {"mode": "random", "width": 1280, "seed": 0, "shape_mode": "match_source"}
        one = control_matrix("record", {"mode": "random", "width": 1280, "seed": 0, "tokens": 1})
        for length in (50, 197, 991):
            full = control_matrix("record", spec, shape=(length, 1280))
            torch.testing.assert_close(one, full[:1], rtol=0, atol=0)
            for fraction in (0.25, 0.5, 0.75):
                partial = control_matrix("record", {**spec, "length_fraction": fraction}, shape=(length, 1280))
                torch.testing.assert_close(partial, full[:len(partial)], rtol=0, atol=0)
                self.assertEqual(partial.untyped_storage().nbytes(), partial.numel() * partial.element_size())

    def test_invalid_length_fractions_fail(self):
        for fraction in (0, -0.25, 1.25, "nan", "inf"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                control_matrix("record", {"mode": "constant", "width": 7, "shape_mode": "match_source", "length_fraction": fraction}, shape=(5, 7))
        with self.assertRaisesRegex(ValueError, "requires shape_mode"):
            control_matrix("record", {"mode": "random", "length_fraction": 0.25})

    def test_fractional_metadata_and_cache_names_distinguish_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root, shape=(9, 7))
            names = set()
            for fraction, tokens in ((0.25, 3), (0.5, 5), (0.75, 7), (1, 9)):
                spec = {**self.matched_spec(root), "length_fraction": fraction}
                payload = self.embed(root, spec)
                self.assertEqual(payload["source_length"], 9)
                self.assertEqual(payload["length"], tokens)
                self.assertEqual(payload["length_fraction"], fraction)
                self.assertEqual(payload["length_rounding"], "ceil_min_one")
                names.add(payload["model_name"])
            self.assertEqual(len(names), 4)
            self.assertEqual(control_model_name(self.matched_spec(root)), control_model_name({**self.matched_spec(root), "length_fraction": 1}))


if __name__ == "__main__":
    unittest.main()
