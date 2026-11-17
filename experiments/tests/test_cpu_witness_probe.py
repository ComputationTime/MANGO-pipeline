import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from probe_cpu_witnesses import candidates


class CandidateTests(unittest.TestCase):
    def test_exact_match_is_the_only_proposed_pair(self):
        q={'q':'ABCDEFXXXXWXYZ'};t={'exact':'ABCDEFXXXXWXYZ','other':'ABCDEFYYYYWXYZ'}
        self.assertEqual(list(candidates(q,t)),[('q','exact')])

    def test_absent_bucket_has_no_candidate_not_a_safety_decision(self):
        self.assertEqual(list(candidates({'q':'ABCDEFXXXXWXYZ'},{'t':'DIFFERENT'})),[])

    def test_bounded_candidates_are_real_and_deterministic(self):
        q={'q':'ABCDEF????WXYZ'};t={str(i):f'ABCDEF{i:04d}WXYZ' for i in range(50)}
        a=list(candidates(q,t,4))
        self.assertEqual(len(a),4);self.assertTrue(all(k in t for _,k in a))
        self.assertEqual(a,list(candidates(q,dict(reversed(list(t.items()))),4)))


if __name__=='__main__':unittest.main()
