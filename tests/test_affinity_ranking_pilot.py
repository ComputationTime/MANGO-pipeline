"""Check the actual batching and controlled affinity-gradient intervention."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow/scripts'))
import torch
from affinity_ranking_pilot import BatchScorer,loss_parts,native_schedule
from affinity_study_common import digest,fingerprint,write_json
from short_affinity_model import build
from controlled_antigen_model import mode
from antigen_diagnostics import iglm_token_nll


class AffinityRankingPilotTests(unittest.TestCase):
    def test_ranking_gradient_has_correct_direction(self):
        native=torch.tensor([2.,3.],requires_grad=True);pair=torch.tensor([3.,1.],requires_grad=True)
        loss,ce,rank=loss_parts(native,pair,.1,.25)
        grad=torch.autograd.grad(loss-ce,pair)[0]
        self.assertGreater(float(grad[0]),0);self.assertLess(float(grad[1]),0)
        baseline,basece,_=loss_parts(native,pair,0.,.25)
        self.assertEqual(float(baseline.detach()),float(basece.detach()))

    def test_schedule_never_uses_heldout_rows(self):
        rows=[dict(id='a',split='train',biological_group='x',example_target_chains='heavy'),
              dict(id='b',split='test',biological_group='y',example_target_chains='light')]
        schedule=native_schedule(rows,5,4,9)
        self.assertTrue(all(x['id']=='a' for batch in schedule for x in batch))
        self.assertEqual(schedule,native_schedule(rows,5,4,9))

    def test_batched_nll_matches_scalar_and_padding_is_ignored(self):
        from iglm_baseline import MODEL_CONFIG
        cfg=dict(MODEL_CONFIG,n_layer=1,n_head=4,n_embd=32,n_inner=64,n_positions=64,n_ctx=64)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'ag.pt';seq='ACDE'
            torch.save(dict(embedding=torch.randn(4,1280),antigen_sha256=fingerprint(seq)),path)
            write_json(root/'embeddings_complete.json',dict(files_sha256={str(path):digest(path)}))
            initial_scores=[]
            for branch in (False,True):
                model=build(dict(seed=3,branch=branch,pool='mean'),config=cfg);mode(model,False)
                scorer=BatchScorer(model,root)
                a=dict(resolved_ag_seq=seq,resolved_H_seq='ACDEFG',resolved_L_seq='ACD')
                b=dict(a,resolved_H_seq='ACD')
                with torch.inference_mode():
                    means,_,counts=scorer.losses([(a,'heavy'),(b,'heavy'),(a,'light')])
                    initial_scores.append(means)
                    for i,(r,c) in enumerate([(a,'heavy'),(b,'heavy'),(a,'light')]):
                        context=scorer.context[fingerprint(seq)][None] if branch else None
                        target=r['resolved_H_seq' if c=='heavy' else 'resolved_L_seq']
                        scalar=iglm_token_nll(model,c,target,h_ag=context).mean()
                        self.assertAlmostEqual(float(means[i]),float(scalar),places=5)
                        self.assertEqual(int(counts[i]),len(target)+1)
            torch.testing.assert_close(initial_scores[0],initial_scores[1],atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
