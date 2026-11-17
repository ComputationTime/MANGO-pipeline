"""Local affinity parsing and repeated-pair aggregation regression tests."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'workflow/scripts'))
from prepare_mango_binding_affinity import finite_affinity, flag
from finalize_openalphaseq_affinity import main as finalize_measurements
from reuse_mango_binding_cache import checked_copy


class LocalBindingTests(unittest.TestCase):
    def test_affinity_and_assay_flag_parsing(self):
        for value in ('', 'nan', 'NaN', 'inf', '-inf', 'None'):
            self.assertIsNone(finite_affinity(value))
        self.assertEqual(finite_affinity('-2.5'), -2.5)
        self.assertEqual(flag('True'), 1)
        self.assertEqual(flag('False'), 0)
        with self.assertRaises(ValueError):
            flag('unknown')

    def test_normalized_pairs_average_only_qualifying_local_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / 'records.csv'; records.write_text('split\ntrain\n')
            study = root / 'study.json'
            study.write_text(json.dumps(dict(experiment_root=str(root), training_records=str(records))))
            with sqlite3.connect(root / 'raw.sqlite') as db:
                db.executescript('''CREATE TABLE observations (antibody_id,antigen_id,affinity,
                    sufficient_replicates,above_background);
                    INSERT INTO observations VALUES ('raw1','ag',1.,1,1), ('raw2','ag',3.,1,0),
                        ('raw1','ag',99.,0,1), ('raw1','ag',NULL,1,1);''')
            with sqlite3.connect(root / 'cohort.sqlite') as db:
                db.executescript('''CREATE TABLE raw_map (raw_id,query_id);
                    INSERT INTO raw_map VALUES ('raw1','q'),('raw2','q');
                    CREATE TABLE queries (id,heavy,light,heavy_novel,light_novel);
                    INSERT INTO queries VALUES ('q','h','l',1,0);
                    CREATE TABLE targets (id,chain,sequence,ntokens,seen_training);
                    INSERT INTO targets VALUES ('h','heavy','AAA',4,0),('l','light','CCC',4,1);''')
            with patch('finalize_openalphaseq_affinity.finalize'):
                finalize_measurements(str(study))
            with sqlite3.connect(root / 'measurements.sqlite') as db:
                row = db.execute('SELECT all_count,all_sum,reliable_count,reliable_sum,positive_count,positive_sum FROM pairs').fetchone()
                self.assertEqual(row, (3,103.,2,4.,1,1.))
                self.assertEqual(db.execute('SELECT target_id FROM required').fetchall(), [('h',)])

    def test_cache_copy_rejects_corruption_and_existing_difference(self):
        from finalize_openalphaseq_pairing import digest
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'; source.write_text('verified')
            destination = Path(directory) / 'copy'
            checked_copy(source, destination, digest(source))
            self.assertEqual(destination.read_text(), 'verified')
            with self.assertRaises(ValueError):
                checked_copy(source, destination, 'wrong-hash')
            destination.write_text('different')
            with self.assertRaises(ValueError):
                checked_copy(source, destination, digest(source))


if __name__ == '__main__':
    unittest.main()
