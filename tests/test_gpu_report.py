import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

from report_gpu_results import report


class GPUReportTests(unittest.TestCase):
    def test_successful_numbers_survive_an_independent_embedder_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            good.mkdir()
            for name in ("weights", "embedding", "checkpoint"):
                (good / name).write_text("ready\n")
            evaluation = {
                "run_id": "good__1234",
                "splits": {
                    "test": {
                        "n_examples": 2,
                        "n_tokens": 20,
                        "nll": 1.25,
                        "perplexity": 3.49,
                    }
                },
            }
            (good / "eval.json").write_text(json.dumps(evaluation))
            with (good / "predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["status"])
                writer.writeheader()
                writer.writerow({"status": "ok"})
                writer.writerow({"status": "error"})

            def spec(tag, directory):
                return {
                    "embedder": tag,
                    "run_id": f"{tag}__1234",
                    "weights": str(directory / "weights"),
                    "embedding": str(directory / "embedding"),
                    "checkpoint": str(directory / "checkpoint"),
                    "evaluation": str(directory / "eval.json"),
                    "predictions": str(directory / "predictions.csv"),
                    "analysis_metrics": {},
                    "logs": {},
                }

            output_json = root / "report.json"
            output_csv = root / "numbers.csv"
            specs = [spec("good", good), spec("broken", root / "broken")]
            (good / "iglm.csv").write_text("score\n1.0\n")
            specs[0]["analysis_metrics"] = {
                "iglm": str(good / "iglm.csv"),
                "germline": str(good / "germline.csv"),
            }
            report(
                specs,
                str(root / "preflight.json"), [], [],
                str(output_json), str(output_csv),
            )
            result = json.loads(output_json.read_text())
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["summary"]["successful_embedders"], ["good"])
            self.assertEqual(
                result["summary"]["failed_or_incomplete_embedders"], ["broken"]
            )
            self.assertEqual(
                result["summary"]["analysis_incomplete_embedders"], ["good"]
            )
            self.assertEqual(
                result["runs"]["good"]["status"],
                "core_complete_analysis_incomplete",
            )
            self.assertEqual(
                result["runs"]["good"]["missing_analysis_metrics"], ["germline"]
            )
            self.assertEqual(result["runs"]["good"]["splits"]["test"]["nll"], 1.25)
            self.assertEqual(result["runs"]["good"]["prediction_counts"]["ok"], 1)
            self.assertEqual(result["runs"]["broken"]["failure_stage"], "weights")
            with output_csv.open() as handle:
                rows = list(csv.DictReader(handle))
            good_row = next(row for row in rows if row["embedder"] == "good")
            self.assertEqual(good_row["nll"], "1.25")


if __name__ == "__main__":
    unittest.main()
