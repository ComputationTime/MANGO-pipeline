"""Prevent tie bias and fabricated experimental negatives in pairing metrics."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from report_openalphaseq_pairing import retrieval, binary_auc, within_query_auc, evaluate_matrix, specificity_control


class PairingMetricTests(unittest.TestCase):
    def test_antigen_free_iglm_gets_expected_chance_with_multiple_positives(self):
        scores = np.ones((2, 4))
        positives = np.array([[1, 0, 0, 0], [0, 1, 0, 1]], dtype=bool)
        valid, top1, mrr = retrieval(scores, positives)
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(top1, [.25, .5])
        np.testing.assert_allclose(mrr, [sum(1/r for r in range(1,5))/4, 13/18])
        # Changing column order cannot improve an antigen-free model.
        perm = [3, 1, 0, 2]
        _, shuffled_top1, shuffled_mrr = retrieval(scores[:,perm], positives[:,perm])
        np.testing.assert_array_equal(top1, shuffled_top1)
        np.testing.assert_array_equal(mrr, shuffled_mrr)

    def test_unknown_pairs_do_not_become_measured_negatives(self):
        scores = np.array([[1., 2., 0.], [3., 1., 2.], [2., 1., 0.]])
        labels = np.array([[1, -1, 0], [1, -1, 0], [1, 0, 0]])
        result = within_query_auc(scores, labels, np.ones(3, dtype=bool))
        self.assertEqual(result['queries'], 2)
        self.assertEqual(result['mean_auc'], .5)

    def test_auroc_orientation_and_ties(self):
        self.assertEqual(binary_auc([0, 0, 1, 1], [0, 1, 2, 3]), 1.)
        self.assertEqual(binary_auc([0, 0, 1, 1], [3, 2, 1, 0]), 0.)
        self.assertEqual(binary_auc([0, 1], [1, 1]), .5)
        self.assertIsNone(binary_auc([1, 1], [0, 1]))

    def test_family_macro_does_not_weight_eight_rbd_proteins_as_eight_families(self):
        families = np.array(['RBD']*8 + ['TIGIT']*2 + ['PD1', 'HER2'])
        labels = np.zeros((4, 12)); labels[np.arange(4), [0,8,10,11]] = 1
        result = evaluate_matrix(np.ones((4,12)), labels, families, np.ones(4, dtype=bool))
        self.assertEqual(result['families_present'], 4)
        self.assertEqual(result['family_balanced_top1'], .25)
        self.assertAlmostEqual(result['protein_top1'], 1/12)

    def test_fixed_antigen_preference_is_not_sequence_specific(self):
        scores = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]])
        labels = np.array([[1,-1], [1,-1], [1,-1], [-1,1]])
        control = specificity_control(scores, labels, np.ones(4, dtype=bool), ['library']*4)
        self.assertEqual(control['observed_auc'], .75)
        self.assertEqual(control['library_mean_auc'], .75)
        self.assertEqual(control['reassigned_mean_auc'], .75)
        self.assertEqual(control['excess_over_reassignment'], 0.)

    def test_sequence_specific_matching_beats_within_library_reassignment(self):
        scores = np.array([[1., 2.], [1., 2.], [2., 1.], [2., 1.]])
        labels = np.array([[1,-1], [1,-1], [-1,1], [-1,1]])
        control = specificity_control(scores, labels, np.ones(4, dtype=bool), ['library']*4, repeats=32)
        self.assertEqual(control['observed_auc'], 1.)
        self.assertEqual(control['library_mean_auc'], .5)
        self.assertGreater(control['excess_over_reassignment'], .5)


if __name__ == '__main__':
    unittest.main()
