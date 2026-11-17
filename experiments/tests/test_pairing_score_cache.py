"""Cached/batched scores must reproduce the trained scalar NLL for every model."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from model_common import MangoModel
from iglm_baseline import IgLMScratch
from antigen_diagnostics import mango_token_nll,iglm_token_nll
from pairing_score_cache import CachedChainScorer
from pairing_prefix_score_cache import SharedPrefixChainScorer


class PairingCacheTests(unittest.TestCase):
    def test_shared_target_prefix_matches_uncached_scores_and_preserves_antigen_cache(self):
        torch.set_num_threads(2); torch.manual_seed(2)
        sequences = ['ACDEFGHIKLMN', 'ACDEFGHIKLMNPQRST', 'ACDEFGHIKLAA']
        cdr3 = [[1,10], [2,13,15], [3,9]]
        for family, architecture in [('mango', MangoModel.ARCHITECTURE),
                ('mango', MangoModel.LEGACY_PREFIX_ARCHITECTURE),
                ('cross_attention', MangoModel.CROSS_ATTENTION_ARCHITECTURE), ('iglm', None)]:
            model = (MangoModel(8, architecture=architecture).eval() if architecture else
                     IgLMScratch(dict(initialization='random_from_scratch', objective='full_chain_autoregressive_v1',
                                      species_token='[HUMAN]')))
            model.lm.eval(); antigens = {'a': torch.randn(1,7,8)}
            original = CachedChainScorer(model, family, antigens)
            optimized = SharedPrefixChainScorer(model, family, antigens)
            for chain in ('heavy', 'light'):
                with self.subTest(family=family, chain=chain), torch.no_grad():
                    expected = original.score('a', chain, sequences, cdr3)
                    got = optimized.score('a', chain, sequences, cdr3)
                    for a,b in zip(got, expected):
                        torch.testing.assert_close(a,b,rtol=1e-5,atol=3e-6)
                    again = optimized.score('a', chain, sequences, cdr3)
                    for a,b in zip(again, got):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    single = optimized.score('a', chain, [sequences[0]])[0]
                    torch.testing.assert_close(single, expected[0][:1], rtol=1e-5, atol=3e-6)

    def test_all_architectures_ragged_batches_both_chains_and_unchanged_prompt_cache(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        antigen=torch.randn(1,7,8)
        sequences=['ACD','EFGHIK','LM']
        architectures=[('mango',MangoModel.ARCHITECTURE),
                       ('mango',MangoModel.LEGACY_PREFIX_ARCHITECTURE),
                       ('cross_attention',MangoModel.CROSS_ATTENTION_ARCHITECTURE)]
        for family,architecture in architectures+[('iglm',None)]:
            model=(MangoModel(8,architecture=architecture).eval() if architecture else
                   IgLMScratch(dict(initialization='random_from_scratch',objective='full_chain_autoregressive_v1',
                                    species_token='[HUMAN]')))
            model.lm.eval()
            scorer=CachedChainScorer(model,family,{'antigen':antigen})
            for chain in ('heavy','light'):
                with self.subTest(family=family,chain=chain),torch.no_grad():
                    totals,cdr,losses=scorer.score('antigen',chain,sequences,[[0,2],[1,4],[0]])
                    original_cache=[t.clone() for layer in scorer.prompt('antigen',chain)['cache'] for t in layer]
                    for i,seq in enumerate(sequences):
                        expected=(mango_token_nll(model,antigen,chain,seq) if architecture else iglm_token_nll(model,chain,seq))
                        torch.testing.assert_close(losses[i,:len(seq)+1],expected,rtol=1e-5,atol=2e-6)
                        torch.testing.assert_close(totals[i],expected.double().sum(),rtol=1e-5,atol=2e-6)
                    repeated=scorer.score('antigen',chain,sequences)[0]
                    torch.testing.assert_close(repeated,totals,rtol=0,atol=0)
                    for before,after in zip(original_cache,[t for layer in scorer.prompt('antigen',chain)['cache'] for t in layer]):
                        torch.testing.assert_close(before,after,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
