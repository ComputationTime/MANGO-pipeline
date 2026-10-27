"""Behavioral checks for antigen memory outside the causal decoder prefix."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts"))
from model_common import MangoModel


class CrossAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.model = MangoModel(3, architecture=MangoModel.CROSS_ATTENTION_ARCHITECTURE).eval()
        self.ag = torch.randn(1, 4, 3)
        self.ids = self.model.target_token_ids("ACD")

    def checkpoint(self):
        return {
            "architecture": self.model.architecture,
            "d_ag": self.model.d_ag,
            "target_chains": list(self.model.target_chains),
            "chain_embedding": self.model.chain_embedding.state_dict(),
            "antigen_projection": self.model.antigen_projection.state_dict(),
            "lm": self.model.lm.state_dict(),
        }

    def test_only_selector_and_bos_enter_prefix(self):
        prefix = self.model._prefix(self.ag, "heavy")
        self.assertEqual(tuple(prefix.shape), (1, 2, 488))
        torch.testing.assert_close(prefix, self.model._prefix(torch.randn(1, 50, 3), "heavy"))
        self.assertTrue(all(hasattr(block, "crossattention") for block in self.model.lm.transformer.h))
        self.assertGreater(self.model.cross_attention_parameters, 0)

    def test_loss_labels_only_target_and_eos_and_preserves_causality(self):
        with patch.object(self.model.lm, "forward", wraps=self.model.lm.forward) as forward:
            loss = self.model.loss(self.ag, "heavy", self.ids)
            kwargs = forward.call_args.kwargs
        self.assertEqual(tuple(kwargs["inputs_embeds"].shape), (1, 6, 488))
        self.assertTrue((kwargs["labels"][:, :2] == -100).all())
        torch.testing.assert_close(kwargs["labels"][:, 2:], self.ids[:, 1:])
        torch.testing.assert_close(kwargs["encoder_hidden_states"], self.model.antigen_projection(self.ag))
        inputs = kwargs["inputs_embeds"]
        context = {"encoder_hidden_states": kwargs["encoder_hidden_states"]}
        with torch.no_grad():
            logits = self.model.lm(inputs_embeds=inputs, **context).logits
            expected = F.cross_entropy(logits[:, 1:-1].reshape(-1, 26), self.ids[:, 1:].reshape(-1))
            changed = inputs.clone()
            changed[:, 2:] = torch.randn_like(changed[:, 2:])
            altered = self.model.lm(inputs_embeds=changed, **context).logits
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(logits[:, :2], altered[:, :2])

    def test_antigen_affects_loss_and_cross_attention_receives_gradients(self):
        antigen = self.ag.clone().requires_grad_()
        loss = self.model.loss(antigen, "heavy", self.ids)
        loss.backward()
        self.assertGreater(antigen.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.antigen_projection.weight.grad.abs().sum().item(), 0)
        for block in self.model.lm.transformer.h:
            self.assertGreater(block.crossattention.c_attn.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.crossattention.q_attn.weight.grad.abs().sum().item(), 0)
        other = self.model.loss(torch.zeros_like(antigen), "heavy", self.ids)
        self.assertGreater(abs(loss.item() - other.item()), 1e-7)

    def test_generation_keeps_antigen_memory_at_every_cached_step_and_batches(self):
        calls = []
        def capture(module, args, kwargs):
            calls.append(kwargs.copy())
        handle = self.model.lm.register_forward_pre_hook(capture, with_kwargs=True)
        self.model.lm.generation_config.min_new_tokens = 3
        try:
            generated = self.model.generate_chain_batch(self.ag, "light", 2, 3, False, 1.0, 1.0)
        finally:
            handle.remove()
        self.assertEqual([len(row) for row in generated], [3, 3])
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(calls[0]["inputs_embeds"], self.model._prefix(self.ag, "light").expand(2, -1, -1))
        expected_memory = self.model.antigen_projection(self.ag).expand(2, -1, -1)
        for call in calls:
            torch.testing.assert_close(call["encoder_hidden_states"], expected_memory)
        self.assertEqual(calls[1]["input_ids"].shape, (2, 1))

    def test_cached_generation_matches_full_causal_recomputation(self):
        generated = self.model.generate_chain(self.ag, "heavy", 3, False, 1.0, 1.0)
        inputs = self.model._prefix(self.ag, "heavy")
        expected = []
        with torch.no_grad():
            for _ in range(3):
                logits = self.model.lm(inputs_embeds=inputs, use_cache=False,
                                       **self.model._antigen_context(self.ag)).logits[:, -1].clone()
                logits[:, [ids[0] for ids in self.model.bad_word_ids]] = -torch.inf
                token = logits.argmax(dim=-1, keepdim=True)
                expected.append(token.item())
                if token.item() == self.model.vocab[">"]:
                    break
                inputs = torch.cat([inputs, self.model.lm.transformer.wte(token)], dim=1)
        self.assertEqual(generated, expected)

    def test_antigen_length_does_not_consume_decoder_positions(self):
        memory = torch.randn(1, self.model.configs.n_positions + 1, 3)
        self.assertTrue(torch.isfinite(self.model.loss(memory, "heavy", self.ids)))
        with self.assertRaisesRegex(ValueError, "exceeding GPT-2 capacity"):
            self.model.generate_chain(self.ag, "heavy", self.model.configs.n_positions, False, 1., 1.)

    def test_checkpoint_dispatch_and_strict_architecture_loading(self):
        checkpoint = self.checkpoint()
        restored = MangoModel.from_checkpoint(checkpoint).eval()
        self.assertEqual(restored.architecture, MangoModel.CROSS_ATTENTION_ARCHITECTURE)
        torch.testing.assert_close(self.model.loss(self.ag, "heavy", self.ids),
                                   restored.loss(self.ag, "heavy", self.ids))
        with self.assertRaises(RuntimeError):
            MangoModel.from_checkpoint({**checkpoint, "architecture": MangoModel.ARCHITECTURE})

    def test_shared_initialization_and_rng_match_prefix_baseline(self):
        torch.manual_seed(13)
        prefix = MangoModel(3)
        prefix_rng = torch.get_rng_state()
        torch.manual_seed(13)
        cross = MangoModel(3, architecture=MangoModel.CROSS_ATTENTION_ARCHITECTURE)
        self.assertTrue(torch.equal(prefix_rng, torch.get_rng_state()))
        for name, weight in prefix.state_dict().items():
            torch.testing.assert_close(weight, cross.state_dict()[name], rtol=0, atol=0)
        difference = sum(p.numel() for p in cross.parameters()) - sum(p.numel() for p in prefix.parameters())
        self.assertEqual(difference, cross.cross_attention_parameters)


if __name__ == "__main__":
    unittest.main()
