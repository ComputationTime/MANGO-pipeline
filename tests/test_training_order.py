import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from training_order import epoch_training_rows


class TrainingOrderTests(unittest.TestCase):
    def test_each_epoch_is_a_distinct_complete_permutation(self):
        rows = list(range(32))
        epoch_zero = epoch_training_rows(rows, seed=13, epoch=0)
        epoch_one = epoch_training_rows(rows, seed=13, epoch=1)

        self.assertCountEqual(epoch_zero, rows)
        self.assertCountEqual(epoch_one, rows)
        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(len(epoch_zero), len(set(epoch_zero)))
        self.assertEqual(len(epoch_one), len(set(epoch_one)))

    def test_order_is_reproducible_and_input_is_not_mutated(self):
        rows = list(range(16))
        original = list(rows)
        first = epoch_training_rows(rows, seed=7, epoch=4)
        second = epoch_training_rows(rows, seed=7, epoch=4)

        self.assertEqual(first, second)
        self.assertEqual(rows, original)

    def test_shuffle_can_be_disabled_explicitly(self):
        rows = list(range(8))
        self.assertEqual(
            epoch_training_rows(rows, seed=0, epoch=99, shuffle=False), rows
        )


if __name__ == "__main__":
    unittest.main()
