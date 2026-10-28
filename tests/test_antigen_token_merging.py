import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow/scripts"))
from embed_antigen_merged import (
    MERGING, antigen_merged, build_merged_embeddings, merge_tokens, merged_model_name,
)


class AntigenTokenMergingTests(unittest.TestCase):
    def spec(self, root, fraction=0.5):
        return dict(width=4, source_embedder="esm2", source_model_name="esm2_t33_650M_UR50D",
                    source_dir=str(root / "esm2"), merging=MERGING, length_fraction=fraction)

    def source(self, root, **overrides):
        path = root / "esm2/train/a.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(id="a", embedder="esm2", model_name="esm2_t33_650M_UR50D",
                       shape=[9, 4], embedding=torch.arange(36).reshape(9, 4).float())
        payload.update(overrides)
        torch.save(payload, path)
        return path

    def test_similarity_merges_nonadjacent_related_tokens(self):
        matrix = torch.tensor([[1., 0.], [0., 1.], [0., 2.], [2., 0.]])
        merged, masses = merge_tokens(matrix, 2)
        torch.testing.assert_close(merged, torch.tensor([[1.5, 0.], [0., 1.5]]))
        torch.testing.assert_close(masses, torch.tensor([[2.], [2.]]))

    def test_exact_lengths_mass_and_weighted_centroid_survive_repeated_merges(self):
        generator = torch.Generator().manual_seed(7)
        for length in (1, 2, 3, 5, 17, 198):
            matrix = torch.randn(length, 4, generator=generator)
            for fraction in (0.25, 0.5, 0.75, 1):
                target = max(1, math.ceil(length * fraction))
                with self.subTest(length=length, fraction=fraction):
                    merged, masses = merge_tokens(matrix, target)
                    self.assertEqual(tuple(merged.shape), (target, 4))
                    self.assertEqual(float(masses.sum()), length)
                    self.assertTrue((masses >= 1).all())
                    torch.testing.assert_close((merged * masses).sum(0), matrix.sum(0), atol=1e-5, rtol=1e-5)

    def test_full_length_identity_and_one_token_mean_match_reusable_endpoints(self):
        matrix = torch.arange(36).reshape(9, 4).float()
        full, _ = merge_tokens(matrix, 9)
        pooled, _ = merge_tokens(matrix, 1)
        self.assertTrue(torch.equal(full, matrix))
        self.assertTrue(torch.equal(pooled, matrix.mean(0, keepdim=True)))
        self.assertNotEqual(full.data_ptr(), matrix.data_ptr())

    def test_zero_vectors_and_similarity_ties_are_finite_and_deterministic(self):
        for matrix in (torch.zeros(9, 4), torch.ones(9, 4)):
            first, masses = merge_tokens(matrix, 3)
            second, repeated_masses = merge_tokens(matrix, 3)
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(masses, repeated_masses))
            self.assertTrue(torch.isfinite(first).all())

    def test_saved_provenance_rounding_and_antibody_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root)
            out = root / "out.pt"
            for fraction, tokens in ((0.25, 3), (0.5, 5), (0.75, 7)):
                spec = self.spec(root, fraction)
                antigen_merged("unused", "a", str(out), "unused", "merged", spec,
                               row={"id": "a", "split": "train", "resolved_H_seq": "IGNORED"})
                first = torch.load(out, weights_only=False)
                antigen_merged("unused", "a", str(out), "unused", "merged", spec,
                               row={"id": "a", "split": "train", "resolved_H_seq": "CHANGED"})
                second = torch.load(out, weights_only=False)
                self.assertTrue(torch.equal(first["embedding"], second["embedding"]))
                self.assertEqual(first["shape"], [tokens, 4])
                self.assertEqual(sum(first["token_masses"]), 9)
                self.assertEqual(first["source_length"], 9)
                self.assertEqual(first["merging"], MERGING)
                self.assertTrue(first["source_antigen_values_used"])

    def test_invalid_sources_and_specs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.spec(root)
            for overrides, message in (
                ({"id": "other"}, "provenance"),
                ({"embedder": "antibody"}, "provenance"),
                ({"model_name": "esm2_t6_8M_UR50D"}, "provenance"),
                ({"shape": [1, 4]}, "invalid"),
                ({"shape": [9, 5], "embedding": torch.ones(9, 5)}, "width"),
                ({"embedding": torch.full((9, 4), float("nan"))}, "finite"),
            ):
                self.source(root, **overrides)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    antigen_merged("unused", "a", str(root / "out.pt"), "unused", "merged", spec,
                                   row={"id": "a", "split": "train"})
            for fraction in (0, -0.1, 1.25, "nan", "inf"):
                with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                    merged_model_name({**spec, "length_fraction": fraction})
            with self.assertRaisesRegex(ValueError, "algorithm"):
                merged_model_name({**spec, "merging": "unknown"})

    def test_batch_resumes_and_refreshes_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            records = root / "records.csv"
            with records.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "split"])
                writer.writeheader()
                writer.writerow(dict(id="a", split="train"))
            output, marker = root / "merged", root / "merged/.batch_complete.json"
            args = (records, output, marker, "merged", self.spec(root), ["train"])
            build_merged_embeddings(*args)
            self.assertEqual(json.loads(marker.read_text())["built"], 1)
            build_merged_embeddings(*args)
            self.assertEqual(json.loads(marker.read_text())["reused"], 1)
            self.source(root, embedding=torch.zeros(9, 4))
            newer = (output / "train/a.pt").stat().st_mtime + 1
            os.utime(source, (newer, newer))
            build_merged_embeddings(*args)
            self.assertEqual(json.loads(marker.read_text())["built"], 1)
            self.assertTrue(torch.equal(torch.load(output / "train/a.pt", weights_only=False)["embedding"], torch.zeros(5, 4)))


if __name__ == "__main__":
    unittest.main()
