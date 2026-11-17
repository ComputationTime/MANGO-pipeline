import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from witness_homology_search import representatives,prefilter_rows,search
from affinity_study_common import fingerprint


class WitnessTests(unittest.TestCase):
    def test_representatives_are_bounded_deterministic_and_real_targets(self):
        targets={'AAAA'+str(i)+'FFFF' for i in range(20)}
        a=representatives(['AAAA1FFFF'],targets,5)
        self.assertEqual(len(a),5);self.assertTrue(a<=targets)
        self.assertEqual(a,representatives(['AAAA1FFFF'],list(reversed(sorted(targets))),5))

    def test_prefilter_rejects_identifiers_outside_the_full_database(self):
        with self.assertRaises(ValueError):prefilter_rows([['q','missing']],{'q':0},{'t':0})

    def test_prefilter_uses_actual_database_keys_and_deduplicates(self):
        self.assertEqual(prefilter_rows([['q','t'],['q','t']],{'q':9},{'t':31}),['9\t31\t0\t0\n'])

    def test_graph_always_bypasses_witness_shortcut(self):
        with patch('witness_homology_search.full_search',return_value={'hit'}) as full:
            result=search({},Path('root'),'all_lineages',{'q'},{'t'},True)
            self.assertEqual(result,{'hit'});full.assert_called_once_with({},Path('root'),'all_lineages',{'q'},{'t'},True)

    def test_unverified_candidates_always_receive_full_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def full(settings,root,name,queries,targets,antibody):
                calls.append((name,set(queries),set(targets)))
                folder=root/'searches'/name;folder.mkdir(parents=True,exist_ok=True)
                # A witness candidate may pass a small-DB E-value but fail full-DB revalidation.
                row=[fingerprint('query'),fingerprint('target'),'1','1','1','1e-9']
                (folder/'hits.tsv').write_text('\t'.join(row)+'\n')
                return {row[0]}
            with patch('witness_homology_search.full_search',side_effect=full),patch('witness_homology_search.verify_candidates',return_value=[]):
                result=search({},root,'external',{'query'},{'target'},True)
            self.assertEqual(result,{fingerprint('query')})
            self.assertEqual(calls[-1],('external_remaining',{'query'},{'target'}))


if __name__=='__main__':unittest.main()
