import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
import model_common as mc
from train_mango import normalize_gradients
from embed_antigen_pyrosetta_pre import isolate_antigen

class ComparisonControls(unittest.TestCase):
    def test_initialization_and_rng_independent_of_embedder_width(self):
        torch.manual_seed(42)
        a = mc.MangoModel(1)
        rng_a = torch.get_rng_state()
        torch.manual_seed(42)
        b = mc.MangoModel(1536)
        self.assertTrue(torch.equal(rng_a, torch.get_rng_state()))
        for x, y in zip(a.lm.parameters(), b.lm.parameters()):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(a.chain_embedding.weight, b.chain_embedding.weight))

    def test_token_gradient_weighting_full_and_partial_windows(self):
        for lengths in ([2, 5, 3], [2]):
            p = torch.nn.Parameter(torch.tensor(0.3))
            targets = [torch.arange(n, dtype=torch.float32) for n in lengths]
            for target in targets:
                ((p-target).square().mean()*len(target)).backward()
            normalize_gradients([p], sum(lengths))
            ref = torch.nn.Parameter(torch.tensor(0.3))
            (ref-torch.cat(targets)).square().mean().backward()
            torch.testing.assert_close(p.grad, ref.grad)

    def test_removes_all_non_antigen_before_scoring(self):
        class Pose:
            chains = ['H', 'A', 'L', 'B', 'X']
            def pdb_info(self): return self
            def chain(self, i): return self.chains[i-1]
            def total_residue(self): return len(self.chains)
            def delete_residue_slow(self, i): self.chains.pop(i-1)
        self.assertEqual(isolate_antigen(Pose(), ['A','B']).chains, ['A','B'])

    def test_old_pre_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'old.pt'
            torch.save({'embedding':torch.ones(2,1), 'model_name':'pyrosetta_ref2015'},p)
            with self.assertRaisesRegex(ValueError, 'Unsafe legacy PRE'):
                mc.load_embedding(p)
