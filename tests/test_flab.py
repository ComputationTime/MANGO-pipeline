"""FLAb average perplexity and the HER2-style binding scatter."""

import csv
import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow" / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"flab_{name}", SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flab = load("flab_common")
plot_flab = load("plot_flab")
HER2 = Path("/Users/tanokri/Desktop/MANGO_BINDING_DATA/HER2_data.csv")


class AveragePerplexityTests(unittest.TestCase):
    def test_paired_score_is_the_mean_of_chain_perplexities(self):
        # mean log-likelihoods whose perplexities are 16 and 20.
        heavy = flab.chain_perplexity_from_log_likelihood(-np.log(16))
        light = flab.chain_perplexity_from_log_likelihood(-np.log(20))
        self.assertAlmostEqual(heavy, 16.0)
        self.assertAlmostEqual(light, 20.0)
        self.assertAlmostEqual(flab.average_perplexity(heavy, light), 18.0)
        # Not exp(length-weighted mean NLL): that would be geometric, not 18.
        weighted_nll = (np.log(16) * 120 + np.log(20) * 108) / 228
        self.assertNotAlmostEqual(np.exp(weighted_nll), 18.0, places=3)

    def test_mango_nll_uses_the_same_chain_perplexity(self):
        heavy = flab.chain_perplexity_from_nll(np.log(12))
        light = flab.chain_perplexity_from_nll(np.log(18))
        self.assertAlmostEqual(flab.average_perplexity(heavy, light), 15.0)

    def test_perfect_perplexity_is_rejected(self):
        with self.assertRaises(ValueError):
            flab.chain_perplexity_from_nll(0.0)
        with self.assertRaises(ValueError):
            flab.chain_perplexity_from_log_likelihood(0.0)
        with self.assertRaises(ValueError):
            flab.average_perplexity(1.0, 8.0)
        with self.assertRaises(ValueError):
            flab.chain_perplexity_from_nll(-0.2)

    def test_invalid_chain_is_not_written_as_a_score(self):
        frame = pd.DataFrame({
            "heavy_sequence": ["A" * 10, "C" * 10],
            "light_sequence": ["D" * 8, "E" * 8],
            "alphaseq_affinity": [2.0, 3.0],
        })
        scored = flab.apply_chain_nlls(
            frame,
            "esm2",
            {"A" * 10: np.log(10), "C" * 10: 0.0},
            {"D" * 8: np.log(14), "E" * 8: np.log(12)},
        )
        self.assertAlmostEqual(scored.loc[0, "esm2_average_perplexity"], 12.0)
        self.assertTrue(np.isnan(scored.loc[1, "esm2_average_perplexity"]))
        self.assertNotEqual(scored.loc[1, "esm2_status"], "ok")


class PrepareTests(unittest.TestCase):
    def test_trastuzumab_splits_off_the_linker(self):
        if not HER2.is_file():
            self.skipTest("AlphaSeq HER2 table is not on this machine")
        with HER2.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        heavy, light = flab.split_scfv(row["mata_sequence"])
        self.assertTrue(heavy.endswith("LVTVSS"))
        self.assertTrue(light.startswith("DIQMTQ"))
        self.assertEqual(heavy + flab.LINKER + light, row["mata_sequence"])


class PlotTests(unittest.TestCase):
    def test_noisy_cloud_is_not_a_perfect_line(self):
        rng = np.random.default_rng(11)
        perplexity = rng.uniform(12.0, 40.0, size=400)
        affinity = 3.2 + 0.04 * perplexity + rng.normal(0.0, 0.85, size=400)
        frame = pd.DataFrame({
            "target": "HER2",
            "heavy_sequence": [f"H{i}" for i in range(400)],
            "light_sequence": [f"L{i}" for i in range(400)],
            "alphaseq_affinity": affinity,
            "iglm_average_perplexity": perplexity,
            "esm2_average_perplexity": rng.uniform(8.0, 30.0, size=400),
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            iglm = plot_flab.plot_average_perplexity(
                frame, "HER2", "iglm", root / "HER2_iglm.png",
            )
            esm2 = plot_flab.plot_average_perplexity(
                frame, "HER2", "esm2", root / "HER2_esm2.png", label="ESM2",
            )
            self.assertLess(abs(iglm["pearson_r"]), 0.95)
            self.assertGreater(iglm["n"], 100)
            self.assertNotAlmostEqual(esm2["pearson_r"], 1.0, places=2)
            for path in (root / "HER2_iglm.png", root / "HER2_esm2.png"):
                data = path.read_bytes()
                self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
                width, height = struct.unpack(">II", data[16:24])
                self.assertGreater(width, 1000)
                self.assertGreater(height, 700)

    def test_perfect_values_cannot_be_plotted(self):
        frame = pd.DataFrame({
            "target": ["HER2"] * 5,
            "alphaseq_affinity": [1, 2, 3, 4, 5],
            "iglm_average_perplexity": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                plot_flab.plot_average_perplexity(
                    frame, "HER2", "iglm", Path(directory) / "bad.png",
                )


if __name__ == "__main__":
    unittest.main()
