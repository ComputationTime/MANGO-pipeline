"""Behavioral checks for antigen-prefix conditioning (model environment)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow' / 'scripts'))
import model_common as mc


class AntigenPrefixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = mc.MangoModel(d_ag=3).eval()
        self.ag = torch.randn(1, 4, 3)
        self.ids = self.model.target_token_ids('ACD')

    def test_loss_is_heavy_and_eos_only_and_causal(self):
        with patch.object(self.model.lm, 'forward', wraps=self.model.lm.forward) as forward:
            loss = self.model.loss(self.ag, "heavy", self.ids)
            kwargs = forward.call_args.kwargs
        prefix_len = 4 + 1 + 1
        labels = kwargs['labels']
        self.assertTrue((labels[:, :prefix_len] == -100).all())
        torch.testing.assert_close(labels[:, prefix_len:], self.ids[:, 1:])
        inputs = kwargs['inputs_embeds']
        with torch.no_grad():
            logits = self.model.lm(inputs_embeds=inputs).logits
            expected = F.cross_entropy(logits[:, prefix_len-1:-1].reshape(-1, 26), self.ids[:, 1:].reshape(-1))
            changed = inputs.clone()
            changed[:, prefix_len:] = torch.randn_like(changed[:, prefix_len:])
            altered = self.model.lm(inputs_embeds=changed).logits
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(logits[:, :prefix_len], altered[:, :prefix_len])

    def test_antigen_changes_predictions_and_receives_gradients(self):
        loss = self.model.loss(self.ag, "heavy", self.ids)
        loss.backward()
        self.assertGreater(self.model.antigen_projection.weight.grad.abs().sum().item(), 0)
        other = self.model.loss(torch.zeros_like(self.ag), "heavy", self.ids)
        self.assertGreater(abs(loss.item() - other.item()), 1e-7)
        self.assertFalse(hasattr(self.model, 'cross_attn'))

    def test_scalar_antigen_is_not_erased(self):
        model = mc.MangoModel(d_ag=1).eval()
        a = model._prefix(torch.zeros(1, 4, 1), "heavy")
        b = model._prefix(torch.ones(1, 4, 1), "heavy")
        self.assertFalse(torch.equal(a, b))

    def test_generation_uses_training_prefix_and_batches(self):
        with patch.object(self.model.lm, 'generate', wraps=self.model.lm.generate) as generate:
            result = self.model.generate_chain_batch(self.ag, "heavy", 2, 3, False, 1.0, 1.0)
            supplied = generate.call_args.kwargs['inputs_embeds']
        torch.testing.assert_close(supplied, self.model._prefix(self.ag, "heavy").expand(2, -1, -1))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(1 <= len(row) <= 3 for row in result))

    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        checkpoint = {'architecture': self.model.ARCHITECTURE, 'd_ag': 3,
                      'antigen_projection': self.model.antigen_projection.state_dict(),
                      'lm': self.model.lm.state_dict(),
                      'target_chains': list(self.model.target_chains),
                      'chain_embedding': self.model.chain_embedding.state_dict()}
        restored = mc.MangoModel.from_checkpoint(checkpoint).eval()
        torch.testing.assert_close(self.model.loss(self.ag, "heavy", self.ids), restored.loss(self.ag, "heavy", self.ids))
        with self.assertRaisesRegex(ValueError, 'fresh training'):
            mc.MangoModel.from_checkpoint({'cross_attn': {}})

    def test_chain_selector_changes_only_selector_and_gets_gradients(self):
        heavy = self.model._prefix(self.ag, "H")
        light = self.model._prefix(self.ag, "L")
        torch.testing.assert_close(heavy[:, :4], light[:, :4])
        torch.testing.assert_close(heavy[:, -1:], light[:, -1:])
        self.assertFalse(torch.equal(heavy[:, 4], light[:, 4]))
        for chain in ("heavy", "light"):
            self.model.loss(self.ag, chain, self.ids).backward()
        self.assertTrue((self.model.chain_embedding.weight.grad.abs().sum(dim=1) > 0).all())

    def test_invalid_or_untrained_chain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown chain"):
            self.model._prefix(self.ag, "species")
        self.model.target_chains = ("heavy",)
        with self.assertRaisesRegex(ValueError, "not trained"):
            self.model._prefix(self.ag, "light")

    def test_old_incompatible_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fresh training"):
            mc.MangoModel.from_checkpoint({"architecture": "antigen_prefix_gpt2_v1"})

    def test_context_overflow_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'exceeding GPT-2 capacity'):
            self.model.generate_chain(self.ag, "heavy", 2048, False, 1.0, 1.0)
        ids = torch.ones(1, 2048, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, 'exceeding GPT-2 capacity'):
            self.model.loss(self.ag, "heavy", ids)


if __name__ == '__main__':
    unittest.main()
