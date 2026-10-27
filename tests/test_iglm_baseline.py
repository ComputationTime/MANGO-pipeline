from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow/scripts"))
from iglm_baseline import IgLMScratch, MODEL_CONFIG, VOCAB, evaluate, experiment_hash, train


SPEC = dict(initialization="random_from_scratch", objective="full_chain_autoregressive_v1",
            species_token="[HUMAN]")
TINY = dict(MODEL_CONFIG, n_layer=1, n_head=2, n_embd=16, n_inner=32,
            n_positions=32, n_ctx=32, attn_pdrop=0, embd_pdrop=0, resid_pdrop=0)


class IgLMBaselineTests(unittest.TestCase):
    def test_full_native_architecture_count_and_random_initialization(self):
        with patch("transformers.GPT2LMHeadModel.from_pretrained", side_effect=AssertionError("pretrained load")):
            torch.manual_seed(0)
            model = IgLMScratch(SPEC)
        self.assertEqual(sum(p.numel() for p in model.lm.parameters()), 12889600)
        self.assertEqual(len(VOCAB), 33)
        self.assertEqual(model.vocab["[HEAVY]"], 31)
        self.assertEqual(model.vocab["[LIGHT]"], 32)
        self.assertEqual(model.vocab["[SEP]"], 3)

    def test_native_prefix_and_loss_cover_only_residues_plus_eos(self):
        model = IgLMScratch(SPEC, TINY)
        model.lm.eval()
        ids, labels = model.encode("ACD", "heavy")
        self.assertEqual(ids.tolist(), [[31, 26, 5, 6, 7, 3]])
        self.assertEqual(labels.tolist(), [[-100, -100, 5, 6, 7, 3]])
        self.assertEqual(int((labels != -100).sum()), len("ACD") + 1)
        light, _ = model.encode("ACD", "light")
        self.assertEqual(light[0, 0].item(), 32)
        logits = model.lm(ids, use_cache=False).logits
        expected = torch.nn.functional.cross_entropy(logits[:, 1:-1].reshape(-1, 33), ids[:, 2:].reshape(-1))
        torch.testing.assert_close(model.loss("ACD", "heavy"), expected)

    def test_rejects_unknown_residues_invalid_protocol_and_excess_length(self):
        model = IgLMScratch(SPEC, TINY)
        for sequence in ("", "AX", "aC", "A" * 30):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                model.encode(sequence, "heavy")
        for override in (dict(initialization="pretrained"), dict(objective="infilling"), dict(species_token="[UNKNOWN]")):
            with self.subTest(override=override), self.assertRaises(ValueError):
                IgLMScratch({**SPEC, **override}, TINY)

    def records(self, root, masked=False):
        path = root / "records.csv"
        rows = [dict(id=split, split=split, ab_ag_cluster=split,
                     resolved_H_seq="ACD", resolved_L_seq="EF", resolved_ag_seq="AG",
                     antigen_chains="A", target_contract="imgt_variable_vh_vl_v1")
                for split in ("train", "val", "test")]
        if masked:
            rows = [dict(row, id=row['id'] + '_' + chain, example_target_chains=chain)
                    for row in rows for chain in ('heavy', 'light')]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_training_evaluation_reload_and_cohort_guard(self):
        with tempfile.TemporaryDirectory() as tmp, patch("device_common.get_device", return_value="cpu"):
            root = Path(tmp)
            records = self.records(root)
            cfg = dict(seed=0, target_chains=["heavy", "light"], train_splits=["train"],
                       val_splits=["val"], lr=1e-3, epochs=1, patience=2, grad_accum=3,
                       loss_plot_interval=100, shuffle_train=True)
            run_dir = root / "iglm_scratch__test"
            train(records, cfg, SPEC, run_dir, "test", {}, model_config=TINY)
            checkpoint = run_dir / "checkpoints/best.pt"
            result = evaluate(records, checkpoint, run_dir / "eval.json")
            self.assertFalse(result["pretrained_weights_loaded"])
            self.assertFalse(result["antigen_conditioning"])
            self.assertFalse(result["antibody_prompt_used"])
            for values in result["splits"].values():
                self.assertEqual(values["n_examples"], 2)
                self.assertEqual(values["n_tokens"], 7)  # 3 + EOS, 2 + EOS
                weighted = sum(v["nll"] * v["n_tokens"] for v in values["by_chain"].values()) / 7
                self.assertAlmostEqual(values["nll"], weighted)
            for filename in ("training_curve.csv", "training_curve.png", "metrics.jsonl", "training_runtime.json"):
                self.assertTrue((run_dir / filename).is_file())
            old = json.loads((run_dir / "model_config.json").read_text())
            self.assertEqual(old["cohort_sha256"], result["cohort_sha256"])
            records.write_text(records.read_text().replace("ACD", "ACF"))
            with self.assertRaisesRegex(ValueError, "cohort differs"):
                evaluate(records, checkpoint, run_dir / "other.json")

    def test_cluster_leakage_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = self.records(root)
            with records.open() as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["ab_ag_cluster"] = rows[0]["ab_ag_cluster"]
            with records.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Cluster crosses splits"):
                train(records, {}, SPEC, root / "run", "test", {})

    def test_transfer_restores_only_weights_and_checks_source_protocol(self):
        with tempfile.TemporaryDirectory() as tmp, patch('device_common.get_device', return_value='cpu'):
            root = Path(tmp)
            records = self.records(root)
            cfg = dict(seed=0, target_chains=['heavy', 'light'], train_splits=['train'],
                       val_splits=['val'], lr=1e-3, epochs=1, patience=2, grad_accum=1,
                       loss_plot_interval=100, shuffle_train=True)
            source = root / 'asd'
            train(records, cfg, SPEC, source, 'asd', {}, model_config=TINY)
            checkpoint = source / 'checkpoints/best.pt'
            original = torch.load(checkpoint, weights_only=False)
            # A distinct target cohort with zero LR makes weight restoration
            # exact while optimizer step counters detect accidental state reuse.
            with records.open() as handle:
                native_rows = list(csv.DictReader(handle))
            for row in native_rows:
                row['id'] = 'native_' + row['id']
            with records.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(native_rows[0]))
                writer.writeheader()
                writer.writerows(native_rows)
            fine = root / 'fine'
            train(records, dict(cfg, seed=7, lr=0), SPEC, fine, 'fine', {},
                  model_config=TINY, initial_checkpoint=checkpoint)
            restored = torch.load(fine / 'checkpoints/best.pt', weights_only=False)
            for key, value in original['lm'].items():
                torch.testing.assert_close(restored['lm'][key], value, rtol=0, atol=0)
            self.assertTrue(all(state['step'].item() == 2 for state in restored['optimizer']['state'].values()))
            result = evaluate(records, fine / 'checkpoints/best.pt', fine / 'eval.json')
            self.assertEqual(result['initialization'], 'asd_checkpoint')
            self.assertTrue(result['pretrained_weights_loaded'])
            self.assertFalse(result['public_pretrained_weights_loaded'])
            self.assertFalse(result['transfer_initialization']['optimizer_loaded'])
            self.assertEqual(result['transfer_initialization']['source_cohort_sha256'], original['metadata']['cohort_sha256'])
            self.assertNotEqual(result['cohort_sha256'], original['metadata']['cohort_sha256'])
            original['metadata']['species_token'] = '[MOUSE]'
            incompatible = root / 'incompatible.pt'
            torch.save(original, incompatible)
            with self.assertRaisesRegex(ValueError, 'incompatible species_token'):
                train(records, cfg, SPEC, root / 'bad', 'bad', {},
                      model_config=TINY, initial_checkpoint=incompatible)

    def test_chain_masks_and_named_heldout_subset(self):
        with tempfile.TemporaryDirectory() as tmp, patch('device_common.get_device', return_value='cpu'):
            root = Path(tmp)
            records = self.records(root, masked=True)
            cfg = dict(seed=0, target_chains=['heavy', 'light'], train_splits=['train'],
                       val_splits=['val'], lr=1e-3, epochs=1, patience=2, grad_accum=1,
                       loss_plot_interval=100, shuffle_train=True)
            run_dir = root / 'masked'
            train(records, cfg, SPEC, run_dir, 'masked', {}, model_config=TINY)
            checkpoint = run_dir / 'checkpoints/best.pt'
            subset = {'test_primary': {'source_split': 'test', 'ids': ['test_heavy', 'test_light']}}
            result = evaluate(records, checkpoint, run_dir / 'eval.json', evaluation_subsets=subset)
            for score in result['splits'].values():
                self.assertEqual(score['n_examples'], 2)
                self.assertEqual(score['n_tokens'], 7)
            self.assertEqual(result['splits']['test_primary'], result['splits']['test'])
            for invalid in ({'test': subset['test_primary']},
                            {'primary': {'source_split': 'test', 'ids': ['unknown']}}):
                with self.assertRaises(ValueError):
                    evaluate(records, checkpoint, run_dir / 'invalid.json', evaluation_subsets=invalid)

    def test_run_identity_ignores_plot_cadence_but_tracks_training_and_species(self):
        cfg = dict(dataset={}, processing={}, experiment={"seed": 0},
                   model=dict(seed=0, lr=1e-4, retrain=True, loss_plot_interval=250),
                   iglm_baseline=dict(SPEC))
        original = experiment_hash(cfg)
        reporting = deepcopy(cfg)
        reporting["model"].update(retrain=False, loss_plot_interval=500)
        self.assertEqual(original, experiment_hash(reporting))
        for section, key, value in (("model", "seed", 1), ("model", "lr", 1e-3),
                                     ("iglm_baseline", "species_token", "[MOUSE]")):
            other = deepcopy(cfg)
            other[section][key] = value
            self.assertNotEqual(original, experiment_hash(other))


if __name__ == "__main__":
    unittest.main()
