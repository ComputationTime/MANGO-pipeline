import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / 'workflow/scripts'
sys.path.insert(0, str(SCRIPTS))
from report_openalphaseq_affinity import paired_nll, pearson
from prepare_openalphaseq_affinity import merge_config, model_directories


class AffinityMetricsTests(unittest.TestCase):
    def test_perplexity_correlation_is_not_nll_correlation(self):
        nll = np.array([0., 1., 2., 3., 4.])
        affinity = np.exp(nll)
        self.assertAlmostEqual(pearson(np.exp(nll), affinity, verify=True), 1.)
        self.assertLess(pearson(nll, affinity, verify=True), .95)

    def test_pair_score_is_token_weighted_before_exponentiating(self):
        combined = paired_nll(np.array([100.]), np.array([100]), np.array([200.]), np.array([50]))
        self.assertAlmostEqual(combined[0], 2.)
        self.assertNotAlmostEqual(np.exp(combined[0]), (np.exp(1.) + np.exp(4.)) / 2)

    def test_single_domain_uses_heavy_loss_only(self):
        self.assertAlmostEqual(paired_nll(120., 120, 0., 0), 1.)

    def test_degenerate_and_nonfinite_measurements(self):
        self.assertIsNone(pearson([1,1,1], [1,2,3]))
        self.assertIsNone(pearson([1,2], [1,2]))
        with self.assertRaises(ValueError):
            pearson([1,2,np.nan], [1,2,3])

    def test_training_identities_match_supported_study(self):
        import yaml
        configuration = yaml.safe_load(Path('experiments/config/openalphaseq_affinity.yaml').read_text())
        models = model_directories(configuration)
        self.assertEqual(set(models), {'one_hot','biopython','pyrosetta_pre','esm2','esm3',
                                      'esm3_structure','esmif','proteinmpnn','iglm'})
        namespace = {'config': merge_config(configuration['training_configs'])}
        source = (SCRIPTS.parent / "rules/common.smk").read_text().split('wildcard_constraints:')[0]
        exec(compile(source, 'workflow/rules/common.smk', 'exec'), namespace)
        for name in set(models) - {'iglm'}:
            self.assertEqual(models[name]['directory'], namespace['run_dir'](name))
        self.assertNotEqual(models['esm3']['directory'], models['esm3_structure']['directory'])
        self.assertEqual(models['iglm']['family'], 'iglm')


if __name__ == '__main__':
    unittest.main()
