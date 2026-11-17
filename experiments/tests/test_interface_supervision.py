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
from interface_supervision_common import graph,average_precision
from interface_supervision_model import InterfaceModel,balanced_contact_loss
from interface_supervision_worker import model_hash
from iglm_baseline import MODEL_CONFIG


class InterfaceSupervisionTests(unittest.TestCase):
    def model(self,branch=True):
        cfg=dict(MODEL_CONFIG,n_layer=1,n_head=4,n_embd=32,n_inner=64,n_positions=64,n_ctx=64)
        return InterfaceModel(4,branch,dict(message_width=8,message_layers=1,contact_width=8),cfg).eval()

    def inputs(self,n=5):
        rng=np.random.default_rng(3);coords=rng.normal(size=(n,3))*4
        idx,edge,valid=graph(coords,np.zeros(n),np.arange(n),'geometry',4)
        ids=torch.tensor([[31,26,5,6,7,3]]);mask=torch.ones_like(ids)
        return ids,mask,torch.randn(1,n,1280),torch.ones(1,n,dtype=torch.long),torch.tensor(idx)[None],torch.tensor(edge)[None],torch.tensor(valid)[None]

    def test_graph_rigid_motion_invariance_and_sequence_control(self):
        rng=np.random.default_rng(8);xyz=rng.normal(size=(17,3));q,_=np.linalg.qr(rng.normal(size=(3,3)))
        a=graph(xyz,np.zeros(17),np.arange(17),'geometry',8);b=graph(xyz@q+100,np.zeros(17),np.arange(17),'geometry',8)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1],b[1],atol=1e-6)
        a=graph(xyz,np.zeros(17),np.arange(17),'sequence',8);b=graph(xyz[::-1],np.zeros(17),np.arange(17),'sequence',8)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_ap_handles_ties_and_absent_positives(self):
        self.assertAlmostEqual(average_precision([1,0,1,0],[4,3,2,1]),5/6)
        self.assertAlmostEqual(average_precision([1,0,1,0],[1,1,1,1]),.5)
        self.assertIsNone(average_precision([0,0],[2,1]))
        self.assertAlmostEqual(average_precision([1,1],[2,1]),1.)

    def test_probe_control_blocks_generator_gradient_but_not_head_gradient(self):
        model=self.model();inputs=self.inputs();model.contact_to_backbone=False
        logits,contacts=model(*inputs);loss=balanced_contact_loss(contacts[0],torch.eye(3,5,dtype=torch.bool));loss.backward()
        self.assertTrue(all(p.grad is None for p in model.lm.parameters()))
        self.assertTrue(all(p.grad is None for p in model.memory.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.contact_head.parameters()),0)
        model.zero_grad(set_to_none=True);model.contact_to_backbone=True;other,contacts=model(*inputs)
        torch.testing.assert_close(logits,other,rtol=0,atol=0)
        balanced_contact_loss(contacts[0],torch.eye(3,5,dtype=torch.bool)).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.lm.parameters() if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.memory.parameters() if p.grad is not None),0)

    def test_shared_decoder_initial_function_and_parameter_matching(self):
        conditional=self.model();unconditional=self.model(False);inputs=self.inputs()
        with torch.inference_mode():
            a,_=conditional(*inputs);b,_=unconditional(*inputs[:2]);torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        self.assertEqual(model_hash(conditional,True),model_hash(unconditional,True))
        gen=sum(p.numel() for p in conditional.lm.parameters())+sum(p.numel() for p in conditional.memory.parameters())
        self.assertLess(abs(sum(p.numel() for p in unconditional.parameters())/gen-1),.001)

    def test_antigen_padding_does_not_change_real_example(self):
        model=self.model();torch.nn.init.normal_(model.lm.transformer.h[0].crossattention.c_proj.weight,std=.02)
        ids,mask,emb,am,idx,edges,valid=self.inputs(5)
        with torch.inference_mode():a,_=model(ids,mask,emb,am,idx,edges,valid)
        padded=torch.randn(1,9,1280);padded[:,:5]=emb;pm=torch.zeros(1,9,dtype=torch.long);pm[:,:5]=1
        pi=torch.zeros(1,9,4,dtype=torch.long);pi[:,:5]=idx;pe=torch.zeros(1,9,4,18);pe[:,:5]=edges;pv=torch.zeros(1,9,4,dtype=torch.bool);pv[:,:5]=valid
        with torch.inference_mode():b,_=model(ids,mask,padded,pm,pi,pe,pv)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_autoregressive_future_target_cannot_change_earlier_predictions(self):
        model=self.model();inputs=list(self.inputs());ids=inputs[0].clone()
        with torch.inference_mode():a,_=model(*inputs)
        ids[0,4]=20;inputs[0]=ids
        with torch.inference_mode():b,_=model(*inputs)
        torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
