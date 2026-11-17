import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from epitope_diagnostics_common import pooling_weights,masks,weighted_difference
from epitope_diagnostics_worker import forward
from controlled_antigen_model import build,mode
from iglm_baseline import MODEL_CONFIG
from antigen_diagnostics import iglm_token_nll


class EpitopeScoringTests(unittest.TestCase):
    def test_exact_pooling_operator_for_short_and_uneven_lengths(self):
        for length in (3,32,124,497):
            values=torch.arange(length,dtype=torch.float64).reshape(1,1,-1)
            expected=torch.nn.functional.adaptive_avg_pool1d(values,32).numpy().ravel()
            actual=pooling_weights(length)@values.numpy().ravel()
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_contact_contrast_controls_global_uniform_effect(self):
        self.assertAlmostEqual(weighted_difference(np.ones(32),np.linspace(0,1,32)),0.)
        self.assertGreater(weighted_difference(np.linspace(0,1,32),np.linspace(0,1,32)),0.)

    def test_token_regions_keep_eos_outside_cdrs_and_framework(self):
        rm=masks(['framework','cdr1','cdr3','eos'])
        self.assertEqual(rm['full'].sum(),4);self.assertEqual(rm['cdrs'].sum(),2)
        self.assertFalse(rm['framework'][-1]);self.assertFalse(rm['cdrs'][-1])

    def test_per_token_batch_scalar_attention_and_memory_permutation(self):
        cfg=dict(MODEL_CONFIG,n_layer=1,n_head=4,n_embd=32,n_inner=64,n_positions=64,n_ctx=64)
        model=build(2,True,config=cfg);model.base_config=cfg;mode(model,False)
        # Nonzero branch output makes permutation invariance meaningful.
        torch.nn.init.normal_(model.lm.transformer.h[0].crossattention.c_proj.weight,std=.02)
        model.lm.set_attn_implementation('eager');contexts=torch.randn(2,32,1280)
        with torch.inference_mode():
            losses,attn=forward(model,['ACDEFG','ACD'],contexts,attention=True)
            ids,_=model.encode('ACDEFG','heavy')
            expected=model.lm(input_ids=ids,encoder_hidden_states=model.antigen_projection(contexts[:1]),output_attentions=True,use_cache=False)
            weights=expected.cross_attentions[0].mean(1)[0,1:8].numpy()
            np.testing.assert_allclose(attn[0],weights,atol=1e-6)
            scalar=iglm_token_nll(model,'heavy','ACDEFG',h_ag=contexts[:1]).numpy()
            np.testing.assert_allclose(losses[0],scalar,atol=1e-6)
            self.assertEqual(attn[0].shape,(7,32));np.testing.assert_allclose(attn[0].sum(1),1,atol=1e-6)
            permuted,_=forward(model,['ACDEFG','ACD'],contexts.roll(7,1))
            np.testing.assert_allclose(losses[0],permuted[0],atol=1e-6)
            absent,_=forward(model,['ACDEFG','ACD'],None)
            self.assertEqual(len(absent[0]),7)


if __name__=='__main__':unittest.main()
