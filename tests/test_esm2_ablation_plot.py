import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflow" / "scripts" / "analysis" / "plot_esm2_ablation.py"
SPEC = importlib.util.spec_from_file_location("plot_esm2_ablation", SCRIPT)
plot_esm2_ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_esm2_ablation)


class ESM2AblationPlotTests(unittest.TestCase):
    def test_plot_reorders_sizes_and_exports_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedders = ["esm2_150m", "esm2_8m", "esm2"]
            specs = {
                "esm2_150m": {"method": "esm2", "size": "t30_150M"},
                "esm2_8m": {"method": "esm2", "size": "t6_8M"},
                "esm2": {"method": "esm2", "size": "t33_650M"},
            }
            widths = {"esm2_8m": 320, "esm2_150m": 640, "esm2": 1280}
            evals = []
            for i, tag in enumerate(embedders):
                path = root / f"ablation-{tag}.json"
                path.write_text(json.dumps({
                    "embedder": tag,
                    "run_id": f"{tag}__hash",
                    "d_ag": widths[tag],
                    "checkpoint_epoch": 2,
                    "splits": {
                        "train": {
                            "nll": 0.5 + i / 10,
                            "perplexity": 1.7,
                            "n_examples": 6,
                            "n_tokens": 60,
                        },
                        "test": {
                            "nll": 0.9 + i / 10,
                            "perplexity": 2.5,
                            "n_examples": 4,
                            "n_tokens": 40,
                        },
                    },
                }))
                evals.append(path)

            figure = root / "fig7.png"
            data_path = root / "fig7.csv"
            plot_esm2_ablation.plot_esm2_ablation(
                evals, embedders, specs, 80, figure, data_path
            )

            data = pd.read_csv(data_path)
            self.assertGreater(figure.stat().st_size, 0)
            self.assertEqual(
                data.drop_duplicates("embedder")["embedder"].tolist(),
                ["esm2_8m", "esm2_150m", "esm2"],
            )
            self.assertEqual(
                data.drop_duplicates("embedder")["esm2_parameters"].tolist(),
                [8_000_000, 150_000_000, 650_000_000],
            )
            self.assertEqual(set(data["split"]), {"train", "test"})


if __name__ == "__main__":
    unittest.main()
