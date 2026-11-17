"""Scientific contract checks for causal, label-free interface routing."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from iglm_baseline import MODEL_CONFIG
from interface_attention_model import InterfaceAttentionModel
from interface_supervision_common import graph
from interface_supervision_worker import model_hash


class InterfaceAttentionTests(unittest.TestCase):
    def model(self,arm='control',**settings):
        cfg=dict(MODEL_CONFIG,n_layer=2,n_head=4,n_embd=32,n_inner=64,n_positions=64,n_ctx=64)
        model=InterfaceAttentionModel(9,arm,dict(message_width=8,message_layers=1,contact_width=8,**settings),cfg).eval()
        # Initial cross-output weights are zero. Activate them so tests can
        # detect a broken routing path or a future-token leak.
        if model.branch:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(46)
                for block in model.lm.transformer.h:
                    torch.nn.init.normal_(block.crossattention.c_proj.weight,std=.1)
        return model

    def inputs(self,n=7):
        rng=np.random.default_rng(6)
        idx,edge,valid=graph(rng.normal(size=(n,3)),np.zeros(n),np.arange(n),'geometry',4)
        generator=torch.Generator().manual_seed(5)
        ids=torch.tensor([[31,26,5,6,7,3]])
        return [ids,torch.ones_like(ids),torch.randn(1,n,1280,generator=generator),
                torch.ones(1,n,dtype=torch.long),torch.tensor(idx)[None],torch.tensor(edge)[None],torch.tensor(valid)[None]]

    def test_matched_initialization_parameters_and_shared_decoder(self):
        models=[self.model(a) for a in ('control','shared_tag','contact_bias')]
        self.assertEqual(len({model_hash(m) for m in models}),1)
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}),1)
        iglm=self.model('iglm')
        # Reactivating cross weights does not alter the shared decoder hash.
        self.assertEqual(model_hash(models[0],True),model_hash(iglm,True))
        self.assertLess(abs(sum(p.numel() for p in iglm.parameters())/sum(p.numel() for p in models[0].parameters())-1),.001)

    def test_routing_off_recovers_same_control_function(self):
        inputs=self.inputs()
        with torch.inference_mode():
            wanted=self.model()(*inputs)[0]
            for arm in ('shared_tag','contact_bias'):
                model=self.model(arm);model.intervention='off'
                actual=model(*inputs)[0]
                torch.testing.assert_close(actual,wanted,atol=0,rtol=0)

    def test_future_residue_cannot_change_earlier_logits_or_next_contact_predictions(self):
        for arm in ('control','shared_tag','contact_bias'):
            model=self.model(arm);inputs=self.inputs()
            with torch.inference_mode():
                logits,contact=model(*inputs)
                inputs[0]=inputs[0].clone();inputs[0][0,3]=20
                other,newcontact=model(*inputs)
            torch.testing.assert_close(logits[:,:3],other[:,:3],atol=0,rtol=0)
            # Contact row 1 predicts the second residue before it is consumed.
            torch.testing.assert_close(contact[:,:2],newcontact[:,:2],atol=0,rtol=0)

    def test_padding_and_capture_preserve_likelihood_in_all_interventions(self):
        for arm in ('shared_tag','contact_bias'):
            model=self.model(arm)
            for variant in ('correct','off','shuffled'):
                model.intervention=variant;inputs=self.inputs()
                with torch.inference_mode():
                    wanted=model(*inputs)[0];model.capture=True
                    captured=model(*inputs)[0]
                    torch.testing.assert_close(wanted,captured,rtol=0,atol=0)
                    torch.testing.assert_close(model.attention.sum(-1),torch.ones(1,3),rtol=1e-6,atol=1e-6)
                    padded=self.inputs(11)
                    padded[2][:,:7]=inputs[2];padded[3][:,7:]=0
                    padded[4][:,:7]=inputs[4];padded[5][:,:7]=inputs[5];padded[6][:,:7]=inputs[6]
                    actual=model(*padded)[0]
                torch.testing.assert_close(actual,wanted,atol=2e-6,rtol=1e-5)

    def test_supervision_reaches_predictor_and_backbone_even_in_control(self):
        model=self.model();inputs=self.inputs()
        logits,_=model(*inputs)
        truth=torch.eye(3,7,dtype=torch.bool)
        model.supervised_contact_loss(0,truth).backward()
        self.assertGreater(float(model.memory.projection.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.lm.transformer.wte.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.lm.transformer.h[0].crossattention.interface_predictor.q.weight.grad.abs().sum()),0)
        # Labels are used only after forward; changing the labels cannot
        # change the generator's already-computed output.
        other,_=model(*inputs)
        torch.testing.assert_close(logits,other,atol=0,rtol=0)

    def test_routed_signal_changes_attention_and_generator(self):
        for arm in ('shared_tag','contact_bias'):
            model=self.model(arm);model.capture=True;inputs=self.inputs()
            with torch.inference_mode():
                correct=model(*inputs)[0];attn=model.attention.clone()
                model.intervention='off';off=model(*inputs)[0]
            self.assertGreater(float((correct-off).abs().max()),1e-6)
            self.assertGreater(float((attn-model.attention).abs().max()),1e-6)


if __name__=='__main__':unittest.main()
