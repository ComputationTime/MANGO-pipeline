"""Protection thresholds and subprocess timeouts cannot silently weaken."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from benchmark_homology_backend import hits,call


class BackendTests(unittest.TestCase):
    def test_boundary_hits_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'hits.tsv';p.write_text('q\tt\t0.900\t0.800\t0.800\t1e-3\n')
            self.assertEqual(len(hits(p)),1)

    def test_each_failed_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'hits.tsv'
            for columns in [('0.899','0.8','0.8','1e-3'),('0.9','0.799','0.8','1e-3'),('0.9','0.8','0.799','1e-3'),('0.9','0.8','0.8','1.001e-3')]:
                p.write_text('q\tt\t'+'\t'.join(columns)+'\n')
                with self.assertRaises(ValueError):hits(p)

    def test_timeout_is_incomplete_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            value=call([sys.executable,'-c','import time; time.sleep(3)'],Path(directory)/'log',.02)
            self.assertEqual(value['status'],'incomplete');self.assertEqual(value['reason'],'time_limit')

    def test_nonzero_exit_is_failed_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            value=call([sys.executable,'-c','raise SystemExit(2)'],Path(directory)/'log',2)
            self.assertEqual(value['status'],'failed');self.assertEqual(value['returncode'],2)


if __name__=='__main__':unittest.main()
