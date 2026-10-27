"""Position resets must retain antigen attention and agree with cached generation."""
from pathlib import Path
import sys
import unittest

import torch
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from fixed_position_prefix import FixedPositionGPT2LMHeadModel, causal_context
from model_common import MangoModel
from antigen_diagnostics import mango_token_nll


class FixedPositionTests(unittest.TestCase):
    def test_production_loss_matches_the_corrected_earlier_variant_and_remains_causal(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        regular = MangoModel(8).eval()
        fixed = MangoModel(8, architecture=MangoModel.FIXED_POSITION_ARCHITECTURE).eval()
        fixed.load_state_dict(regular.state_dict())
        antigen = torch.randn(1, 7, 8, requires_grad=True)
        for chain in ('heavy', 'light'):
            scores = mango_token_nll(fixed, antigen, chain, 'ACD')
            expected = mango_token_nll(regular, antigen, chain, 'ACD', rebase=True)
            torch.testing.assert_close(scores, expected, rtol=0, atol=1e-6)
            torch.testing.assert_close(scores.mean(), fixed.loss(antigen, chain, fixed.target_token_ids('ACD')))
            objective, nll = fixed.loss(antigen, chain, fixed.target_token_ids('ACD'),
                                        training_loss={'name': 'cross_entropy'}, return_nll=True)
            torch.testing.assert_close(objective, scores.mean())
            torch.testing.assert_close(nll, scores.mean())
            later_change = mango_token_nll(fixed, antigen, chain, 'ACE')
            torch.testing.assert_close(scores[:2], later_change[:2], rtol=0, atol=1e-6)
            scores.sum().backward()
        self.assertGreater(float(antigen.grad.abs().sum()), 1e-5)

    def test_checkpoint_restores_the_fixed_architecture_and_generation_policy(self):
        torch.set_num_threads(2)
        model = MangoModel(8, architecture=MangoModel.FIXED_POSITION_ARCHITECTURE).eval()
        checkpoint = dict(architecture=model.architecture, task=model.TASK, d_ag=8,
            target_chains=['heavy', 'light'], lm=model.lm.state_dict(),
            chain_embedding=model.chain_embedding.state_dict(), antigen_projection=model.antigen_projection.state_dict())
        restored = MangoModel.from_checkpoint(checkpoint).eval()
        self.assertTrue(restored.uses_fixed_positions)
        self.assertFalse(restored.uses_cross_attention)
        antigen = torch.randn(1, 5, 8)
        with torch.no_grad():
            torch.testing.assert_close(mango_token_nll(restored, antigen, 'heavy', 'ACD'),
                                       mango_token_nll(model, antigen, 'heavy', 'ACD'), rtol=0, atol=0)
            batches = restored.generate_chain_batch(antigen, 'heavy', batch_size=2,
                max_new_tokens=3, do_sample=False, top_p=1., temperature=1.)
            self.assertEqual(len(batches), 2)
            self.assertTrue(all(1 <= len(ids) <= 3 for ids in batches))

    def test_cached_generation_matches_full_forward_at_every_target_position(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        model = FixedPositionGPT2LMHeadModel(GPT2Config(vocab_size=26, n_positions=64, n_embd=32,
            n_layer=1, n_head=4, bos_token_id=None, eos_token_id=None, pad_token_id=21)).eval()
        for antigen_length in (3, 7):
            with self.subTest(antigen_length=antigen_length), torch.no_grad():
                seed = torch.randn(2, antigen_length + 2, 32)
                context = causal_context(antigen_length, seed)
                output = model.generate(inputs_embeds=seed, attention_mask=context['attention_mask'],
                    antigen_prefix_length=antigen_length, max_new_tokens=4, do_sample=False,
                    return_dict_in_generate=True, output_logits=True)
                self.assertEqual(tuple(output.sequences.shape), (2, 4))
                for step, logits in enumerate(output.logits):
                    inputs = torch.cat([seed, model.transformer.wte(output.sequences[:, :step])], dim=1)
                    expected = model(inputs_embeds=inputs, use_cache=False,
                                     **causal_context(antigen_length, inputs)).logits[:, -1]
                    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
