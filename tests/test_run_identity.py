"""Aggregation must not change training identity or invalidate completed runs."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workflow/scripts'))


def common(config):
    namespace = {'config': config}
    source = (ROOT / 'workflow/rules/common.smk').read_text().split('wildcard_constraints:')[0]
    exec(compile(source, 'workflow/rules/common.smk', 'exec'), namespace)
    return namespace


class RunIdentityTests(unittest.TestCase):
    def test_model_provenance_is_independent_of_aggregation_but_tracks_scientific_changes(self):
        config = yaml.safe_load((ROOT / 'config/config.yaml').read_text())
        single = common(deepcopy(config))
        config['active_embedders'] = ['one_hot', 'biopython', 'esm2', 'esm3_structure']
        aggregate = common(deepcopy(config))
        self.assertEqual(single['run_id']('one_hot'), aggregate['run_id']('one_hot'))
        self.assertEqual(single['model_global_state_for_tag']('one_hot'),
                         aggregate['model_global_state_for_tag']('one_hot'))
        config['model']['epochs'] += 1
        changed = common(config)
        self.assertNotEqual(single['run_id']('one_hot'), changed['run_id']('one_hot'))
        self.assertNotEqual(single['model_global_state_for_tag']('one_hot'),
                            changed['model_global_state_for_tag']('one_hot'))


if __name__ == '__main__':
    unittest.main()
