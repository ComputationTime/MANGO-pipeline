import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

from training_curve import write_training_plot


class TrainingCurveTests(unittest.TestCase):
    def test_writes_iteration_axis_png_with_validation_points(self):
        history = [
            {"iteration": i, "epoch": 0, "phase": "train", "loss": 4 / i}
            for i in range(1, 21)
        ]
        history.append(
            {"iteration": 20, "epoch": 0, "phase": "validation", "loss": 0.3}
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "training_curve.png"
            write_training_plot(history, output, "one_hot__test")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertFalse((output.parent / ".training_curve.tmp.png").exists())


if __name__ == "__main__":
    unittest.main()

