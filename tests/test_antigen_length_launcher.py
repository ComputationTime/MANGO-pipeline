"""Exercise length-sweep orchestration with stubbed GPU/Conda/Snakemake tools."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TAGS = [
    "control_constant_l1", "control_constant_l25", "control_constant_l50",
    "control_constant_l75", "control_constant", "control_random_l1",
    "control_random_l25", "control_random_l50", "control_random_l75", "control_random",
    "control_pooled_esm2", "control_merged_esm2_l25", "control_merged_esm2_l50", "control_merged_esm2_l75",
]


class AntigenLengthLauncherTests(unittest.TestCase):
    def launch(self, root, fail_tag="", mode="antigen-length-ablation", fail_target=""):
        launcher = root / "run_gpu.sh"
        launcher.write_text((ROOT / "run_gpu.sh").read_text())
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("nvidia-smi", "conda"):
            path = bin_dir / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        driver = root / ".snakemake/gpu-driver/bin"
        driver.mkdir(parents=True)
        snakemake = driver / "snakemake"
        # Absolute interpreter path keeps the stub independent of the Conda PATH.
        import sys
        snakemake.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['SWEEP_CALLS'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "failure = 'active_embedders=[' + os.environ['SWEEP_FAIL_TAG'] + ']'\n"
            "sys.exit(1 if failure in sys.argv or os.environ['SWEEP_FAIL_TARGET'] in sys.argv else 0)\n"
        )
        snakemake.chmod(0o755)
        source_eval = root / "artifacts/runs/esm2__c49369e9/eval.json"
        source_eval.parent.mkdir(parents=True)
        source_eval.write_text("{}")
        (root / "artifacts/data/sabdab2_v0.1.0/embeddings/antigen/esm2").mkdir(parents=True)
        figure_data = root / "artifacts/analysis/figures/fig9_antigen_context_length_ablation_data.csv"
        figure_data.parent.mkdir(parents=True)
        figure_data.write_text("existing-comparison\n")
        calls = root / "calls.jsonl"
        env = {
            **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CONDA_EXE": str(bin_dir / "conda"),
            "HF_HOME": str(root / "artifacts/hf"),
            "TORCH_HOME": str(root / "artifacts/torch"),
            "CONDA_PKGS_DIRS": str(root / ".snakemake/pkgs"),
            "SWEEP_CALLS": str(calls), "SWEEP_FAIL_TAG": fail_tag,
            "SWEEP_FAIL_TARGET": fail_target,
        }
        result = subprocess.run(["bash", str(launcher), mode],
                                env=env, capture_output=True, text=True, timeout=30)
        return result, [json.loads(line) for line in calls.read_text().splitlines()]

    def test_all_variants_run_separately_then_aggregate_without_retraining_esm2(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(Path(tmp))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), len(TAGS) + 3)  # preflight, evaluations, IgLM, plot
        for tag, args in zip(TAGS, calls[1:1 + len(TAGS)]):
            self.assertIn(f"active_embedders=[{tag}]", args)
            self.assertEqual(args[-2:], ["--", "evaluate"])
            self.assertIn("code", args)
            self.assertIn("config/antigen_length_ablation.yaml", args)
        self.assertEqual(calls[-1][-1], "antigen_length_ablation")
        self.assertEqual(calls[-2][-1], "iglm_baseline")
        self.assertIn("fig9_antigen_context_length_ablation_data.csv", result.stdout)

    def test_one_failure_does_not_block_other_lengths_or_publish_strict_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(Path(tmp), "control_constant_l25")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), len(TAGS) + 2)
        for tag, args in zip(TAGS, calls[1:]):
            self.assertIn(f"active_embedders=[{tag}]", args)
        self.assertFalse(any(args[-1] == "antigen_length_ablation" for args in calls))

    def test_focused_iglm_mode_trains_then_plots_without_mango_evaluation_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(Path(tmp), mode="iglm-baseline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertEqual([args[-1] for args in calls], ["gpu_preflight", "iglm_baseline", "antigen_length_ablation"])
        self.assertTrue(all("config/iglm_baseline.yaml" in args for args in calls))
        self.assertFalse(any(arg.startswith("active_embedders=") for args in calls for arg in args))

    def test_failed_iglm_does_not_publish_the_focused_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(Path(tmp), mode="iglm-baseline", fail_target="iglm_baseline")
        self.assertEqual(result.returncode, 1)
        self.assertEqual([args[-1] for args in calls], ["gpu_preflight", "iglm_baseline"])


if __name__ == "__main__":
    unittest.main()
