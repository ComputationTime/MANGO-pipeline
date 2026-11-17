"""run_gpu.sh argument handling: flags map onto the internal mode names."""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(*args):
    result = subprocess.run(['bash', str(ROOT / 'run_gpu.sh'), *args], capture_output=True, text=True,
                            env=dict(os.environ, MANGO_PRINT_MODE='1'), timeout=30)
    return result.returncode, result.stdout.strip(), result.stderr


class LauncherModeTests(unittest.TestCase):
    def test_default_is_smoke(self):
        self.assertEqual(resolve()[:2], (0, 'smoke'))

    def test_flags_compose_into_internal_modes(self):
        cases = {('smoke', '--esm3'): 'smoke-esm3', ('study', '--pyrosetta'): 'study-pyrosetta',
                 ('smoke', '--all'): 'smoke-all', ('small', '--esm3', '--pyrosetta'): 'small-all',
                 ('--all', 'weights'): 'weights-all'}
        for args, expected in cases.items():
            with self.subTest(args=args):
                self.assertEqual(resolve(*args)[:2], (0, expected))

    def test_legacy_spellings_still_work(self):
        for mode in ('smoke-all', 'study-esm3', 'weights-pyrosetta', 'train-all', 'iglm-baseline'):
            with self.subTest(mode=mode):
                self.assertEqual(resolve(mode)[:2], (0, mode))

    def test_flags_rejected_for_non_gated_modes_and_unknown_options(self):
        self.assertEqual(resolve('esm2-ablation', '--all')[0], 2)
        self.assertEqual(resolve('smoke', '--bogus')[0], 2)
        self.assertEqual(resolve('smoke', 'study')[0], 2)
        self.assertEqual(resolve('not-a-mode')[0], 2)

    def test_help_lists_modes_and_flags(self):
        code, out, _ = resolve('--help')
        self.assertEqual(code, 0)
        for text in ('doctor', '--esm3', '--pyrosetta', '--all', 'train-all'):
            self.assertIn(text, out)


if __name__ == '__main__':
    unittest.main()
