import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))

from plot_antigen_controls import plot_antigen_controls


class AntigenControlPlotTests(unittest.TestCase):
    def test_plot_exports_all_models_and_splits(self):
        tags = [
            "control_constant",
            "control_random",
            "control_pooled_esm2",
            "control_shuffled",
            "esm2",
        ]
        labels = {
            "control_constant": "Constant (no antigen)",
            "control_random": "Random (no antigen)",
            "control_pooled_esm2": "Mean-pooled ESM2",
            "control_shuffled": "Shuffled ESM2",
            "esm2": "ESM2",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluations = []
            for index, tag in enumerate(tags):
                path = root / f"{tag}.json"
                path.write_text(json.dumps({
                    "embedder": tag,
                    "run_id": f"{tag}__run",
                    "checkpoint_epoch": index,
                    "d_ag": 1280,
                    "splits": {
                        "train": {
                            "nll": 0.3 + index / 100,
                            "perplexity": 1.3,
                            "n_examples": 10,
                            "n_tokens": 100,
                        },
                        "test": {
                            "nll": 0.8 + index / 100,
                            "perplexity": 2.2,
                            "n_examples": 4,
                            "n_tokens": 40,
                        },
                    },
                }))
                evaluations.append(path)
            figure = root / "controls.png"
            data = root / "controls.csv"
            plot_antigen_controls(
                evaluations, tags, labels, 100, figure, data
            )
            self.assertTrue(figure.is_file())
            rows = data.read_text().splitlines()
            self.assertEqual(len(rows), 1 + 2 * len(tags))
            self.assertIn("control_shuffled", data.read_text())
            self.assertIn("ESM2", data.read_text())


if __name__ == "__main__":
    unittest.main()
