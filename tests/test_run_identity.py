"""Aggregation must not change training identity or invalidate completed runs."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'lib'))
import pipeline as pl  # noqa: E402


class RunIdentityTests(unittest.TestCase):
    def test_run_id_is_independent_of_selection_but_tracks_scientific_changes(self):
        config = pl.load_config(ROOT / 'config' / 'config.yaml')
        single = pl.Pipeline(deepcopy(config))
        config['active_embedders'] = ['one_hot', 'esm2', 'esm3_structure']
        aggregate = pl.Pipeline(deepcopy(config))
        self.assertEqual(single.run_id('one_hot'), aggregate.run_id('one_hot'))
        config['model']['epochs'] += 1
        changed = pl.Pipeline(config)
        self.assertNotEqual(single.run_id('one_hot'), changed.run_id('one_hot'))

    def test_select_embedders_flags_and_all(self):
        import argparse
        pipe = pl.Pipeline(pl.load_config(ROOT / 'config' / 'config.yaml'))
        args = argparse.Namespace(embedders=['esm2'], all_embedders=False,
                                  flag_embedders=['one_hot'])
        self.assertEqual(pipe.select_embedders(args), ['esm2', 'one_hot'])
        args = argparse.Namespace(embedders=None, all_embedders=True, flag_embedders=None)
        self.assertEqual(set(pipe.select_embedders(args)),
                         {'one_hot', 'biopython', 'pyrosetta_pre', 'esm2', 'esm3',
                          'esm3_structure', 'esmif', 'proteinmpnn'})
        with self.assertRaises(SystemExit):
            pipe.select_embedders(argparse.Namespace(
                embedders=['nope'], all_embedders=False, flag_embedders=None))


if __name__ == '__main__':
    unittest.main()
