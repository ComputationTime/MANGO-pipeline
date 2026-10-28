import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow/scripts/analysis"))
from plot_antigen_length_ablation import plot_antigen_length_ablation


class AntigenLengthPlotTests(unittest.TestCase):
    def fixture(self, root):
        def split(n_records, nll):
            return {
                "nll": nll, "perplexity": 2.5, "n_examples": 2 * n_records,
                "n_tokens": 9 * n_records,
                "by_chain": {
                    "heavy": {"n_examples": n_records, "n_tokens": 5 * n_records},
                    "light": {"n_examples": n_records, "n_tokens": 4 * n_records},
                },
            }
        common = {
            "cohort_sha256": "same-cohort", "d_ag": 1280,
            "loss_reduction": "target_token_mean_including_eos",
            "architecture": "antigen_chain_gpt2_v3", "antigen_in_decoder_prefix": True,
            "checkpoint_epoch": 2,
            "splits": {"train": split(2, 0.8), "test": split(1, 1.0)},
        }
        reference = root / "esm2.json"
        reference.write_text(json.dumps({**common, "embedder": "esm2", "run_id": "esm2-reference"}))
        tags, specs, paths = [], {}, []
        for mode in ("constant", "random", "esm2_merged"):
            for index, fraction in enumerate((None, 0.25, 0.5, 0.75, 1)):
                tag = f"{mode}_{index}"
                tags.append(tag)
                specs[tag] = {"method": "antigen_control", "mode": mode, "width": 1280}
                if fraction is not None:
                    specs[tag].update(shape_mode="match_source", length_fraction=fraction)
                if mode == "esm2_merged":
                    if fraction is None:
                        specs[tag] = {"method": "antigen_pooled", "pooling": "mean_over_antigen_tokens_v1", "output_tokens": 1}
                    elif fraction == 1:
                        tag = "esm2"
                        tags[-1] = tag
                        specs[tag] = {"method": "esm2", "size": "t33_650M"}
                    else:
                        specs[tag].update(method="antigen_merged", merging="cosine_bipartite_weighted_v1")
                payload = {**deepcopy(common), "embedder": tag, "run_id": tag}
                for values in payload["splits"].values():
                    values["nll"] += index * 0.05
                    n_records = values["by_chain"]["heavy"]["n_examples"]
                    counts = [1 if fraction is None else math.ceil(length * fraction)
                              for length in (9, 13)[:n_records]]
                    values["antigen_context_tokens"] = {
                        "n_records": n_records,
                        "min": min(counts), "max": max(counts),
                        "mean": sum(counts) / len(counts),
                    }
                if tag == "esm2":
                    # Reuse the historical full-length evaluation unchanged.
                    paths.append(reference)
                    continue
                path = root / f"{tag}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
        return paths, reference, tags, specs

    def test_plot_exports_both_splits_all_lengths_and_actual_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            figure, table = root / "lengths.png", root / "lengths.csv"
            plot_antigen_length_ablation(*args, 100, figure, table)
            self.assertTrue(figure.is_file())
            with table.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 30)
            self.assertEqual({row["split"] for row in rows}, {"train", "test"})
            self.assertEqual({row["context_label"] for row in rows}, {"1 token", "¼L", "½L", "¾L", "L"})
            self.assertTrue(all(row["context_tokens_mean"] for row in rows))
            self.assertTrue(all(row["esm2_reference_run_id"] == "esm2-reference" for row in rows))
            merged = [row for row in rows if row["mode"] == "esm2_merged"]
            self.assertEqual(len(merged), 10)
            self.assertEqual({row["token_merging"] for row in merged},
                             {"identity", "mean_over_antigen_tokens_v1", "cosine_bipartite_weighted_v1"})
            self.assertTrue(all(float(row["nll"]) == float(row["esm2_reference_nll"])
                                for row in merged if row["context_label"] == "L"))

    def test_incomparable_or_incomplete_evaluations_are_rejected(self):
        mutations = (
            (lambda payload: payload.update(cohort_sha256="different-cohort"), "fingerprint"),
            (lambda payload: payload.update(architecture="cross-attention"), "prefix architecture"),
            (lambda payload: payload.update(d_ag=640), "width"),
            (lambda payload: payload["splits"]["test"].update(n_tokens=100), "n_tokens"),
            (lambda payload: payload["splits"]["train"].pop("antigen_context_tokens"), "re-evaluate"),
            (lambda payload: payload["splits"]["test"].update(nll=float("nan")), "non-finite"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.fixture(root)
                payload = json.loads(args[0][0].read_text())
                mutate(payload)
                args[0][0].write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    plot_antigen_length_ablation(*args, 100, root / "plot.png", root / "data.csv")
                self.assertFalse((root / "plot.png").exists())

    def test_missing_context_level_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, reference, tags, specs = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "all five context levels"):
                plot_antigen_length_ablation(paths[:-1], reference, tags[:-1], specs,
                                             100, root / "plot.png", root / "data.csv")

    def test_actual_merged_lengths_must_match_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            payload = json.loads(args[0][-4].read_text())
            payload["splits"]["train"]["antigen_context_tokens"]["mean"] = 3.25
            args[0][-4].write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "context lengths differ"):
                plot_antigen_length_ablation(*args, 100, root / "plot.png", root / "data.csv")

    def test_historical_reference_architecture_is_verified_from_model_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            payload = json.loads(args[1].read_text())
            payload.pop("architecture")
            payload.pop("antigen_in_decoder_prefix")
            args[1].write_text(json.dumps(payload))
            contract = root / "model_config.json"
            contract.write_text(json.dumps({"architecture": "antigen_chain_gpt2_v3"}))
            plot_antigen_length_ablation(*args, 100, root / "plot.png", root / "data.csv")
            contract.write_text(json.dumps({"architecture": "cross-attention"}))
            with self.assertRaisesRegex(ValueError, "prefix architecture"):
                plot_antigen_length_ablation(*args, 100, root / "plot.png", root / "data.csv")

    def test_esm2_endpoint_cannot_use_another_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, reference, tags, specs = self.fixture(root)
            payload = json.loads(reference.read_text())
            payload["run_id"] = "different-esm2-run"
            other = root / "other-esm2.json"
            other.write_text(json.dumps(payload))
            paths[-1] = other
            with self.assertRaisesRegex(ValueError, "reuse the reference"):
                plot_antigen_length_ablation(paths, reference, tags, specs, 100,
                                             root / "plot.png", root / "data.csv")

    def iglm_fixture(self, root, reference):
        payload = json.loads(reference.read_text())
        payload.update(embedder="iglm_scratch", architecture="iglm_gpt2_autoregressive_v1",
                       initialization="random_from_scratch", pretrained_weights_loaded=False,
                       antigen_conditioning=False, antibody_prompt_used=False,
                       objective="full_chain_autoregressive_v1", target_contract="imgt_variable_vh_vl_v1",
                       model_config=dict(n_layer=4, n_head=8, n_embd=512, n_positions=512, vocab_size=33),
                       trainable_parameters=12889600, species_token="[HUMAN]", run_id="iglm_scratch__test")
        path = root / "iglm.json"
        path.write_text(json.dumps(payload))
        return path

    def test_iglm_exports_one_antigen_free_reference_per_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            iglm = self.iglm_fixture(root, args[1])
            plot_antigen_length_ablation(*args, 100, root / "plot.png", root / "data.csv", iglm_eval=iglm)
            with (root / "data.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            baseline = [row for row in rows if row["embedder"] == "iglm_scratch"]
            self.assertEqual(len(baseline), 2)
            self.assertEqual({row["split"] for row in baseline}, {"train", "test"})
            self.assertTrue(all(row["model_family"] == "IgLM" and row["context_label"] == "No antigen"
                                and float(row["context_tokens_mean"]) == 0 for row in baseline))

    def test_iglm_protocol_cohort_and_count_mismatches_are_rejected(self):
        for mutate, message in (
            (lambda p: p.update(cohort_sha256="other"), "fingerprint"),
            (lambda p: p.update(initialization="pretrained"), "protocol"),
            (lambda p: p.update(objective="infilling"), "protocol"),
            (lambda p: p.update(trainable_parameters=500), "architecture"),
            (lambda p: p["splits"]["test"].update(n_tokens=42), "counts"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.fixture(root)
                iglm = self.iglm_fixture(root, args[1])
                payload = json.loads(iglm.read_text())
                mutate(payload)
                iglm.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    plot_antigen_length_ablation(*args, 100, root / "plot.png", root / "data.csv", iglm_eval=iglm)


if __name__ == "__main__":
    unittest.main()
