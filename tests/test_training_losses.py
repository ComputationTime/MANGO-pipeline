"""Focal probability/gradient behavior and antigen-only causal score masking."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow' / 'scripts'))
from training_losses import causal_token_losses, loss_configuration
from model_common import MangoModel


class FocalLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_focal_matches_definition_and_differentiates_focusing_weight(self):
        logits = torch.tensor([[[2., -1., .2], [-.5, 1., 0.], [.4, .1, -.4], [1., 1., 1.]]], requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 0]])
        objective, nll = causal_token_losses(logits, labels, {'name': 'focal', 'gamma': 2})
        log_pt = F.log_softmax(logits[0, 1:3], -1).gather(1, torch.tensor([[2], [0]])).squeeze(1)
        expected = (-log_pt * (1 - log_pt.exp()).pow(2)).mean()
        torch.testing.assert_close(nll, -log_pt.mean())
        torch.testing.assert_close(objective, expected)
        actual_gradient = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(actual_gradient[0, 0].abs().sum().item(), 0)
        self.assertEqual(actual_gradient[0, 3].abs().sum().item(), 0)

    def test_gamma_zero_is_ce_and_easy_tokens_are_downweighted(self):
        logits = torch.tensor([[[8., 0.], [0., 0.], [0., 0.]]])
        labels = torch.tensor([[-100, 0, 1]])
        baseline, nll = causal_token_losses(logits, labels)
        zero, _ = causal_token_losses(logits, labels, {'name': 'focal', 'gamma': 0})
        focal, _ = causal_token_losses(logits, labels, {'name': 'focal', 'gamma': 2})
        torch.testing.assert_close(zero, baseline)
        torch.testing.assert_close(nll, F.cross_entropy(logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1)))
        self.assertLess(focal.item(), nll.item())

    def test_extreme_probabilities_have_finite_loss_and_gradients(self):
        logits = torch.tensor([[[1000., -1000.], [-1000., 1000.], [0., 0.]]], requires_grad=True)
        objective, nll = causal_token_losses(logits, torch.tensor([[-100, 0, 0]]), {'name': 'focal'})
        objective.backward()
        self.assertTrue(torch.isfinite(objective) and torch.isfinite(nll))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_both_architectures_keep_original_nll_and_score_target_plus_eos(self):
        for architecture in (MangoModel.ARCHITECTURE, MangoModel.CROSS_ATTENTION_ARCHITECTURE):
            torch.manual_seed(1)
            model = MangoModel(3, architecture=architecture).eval()
            antigen = torch.randn(1, 4, 3)
            target = model.target_token_ids('ACD')
            baseline = model.loss(antigen, 'heavy', target)
            objective, nll = model.loss(antigen, 'heavy', target, training_loss={'name': 'focal'}, return_nll=True)
            zero = model.loss(antigen, 'heavy', target, training_loss={'name': 'focal', 'gamma': 0})
            torch.testing.assert_close(nll, baseline)
            torch.testing.assert_close(zero, baseline)
            self.assertEqual(model.n_target_tokens(target), 4)
            objective.backward()
            self.assertGreater(model.antigen_projection.weight.grad.abs().sum().item(), 0)
            if model.uses_cross_attention:
                self.assertGreater(sum(p.grad.abs().sum().item() for n, p in model.lm.named_parameters()
                                       if '.crossattention.' in n and p.grad is not None), 0)

    def test_invalid_loss_parameters_and_empty_target_are_rejected(self):
        for value in ({'name': 'dice'}, {'name': 'focal', 'gamma': -1}, {'name': 'focal', 'gamma': float('nan')},
                      {'name': 'cross_entropy', 'gamma': 2}):
            with self.assertRaises(ValueError):
                loss_configuration(value)
        with self.assertRaisesRegex(ValueError, 'no scored'):
            causal_token_losses(torch.zeros(1, 3, 2), torch.full((1, 3), -100))


if __name__ == '__main__':
    unittest.main()
