"""Leakage and assay-label checks for the external antigen-pairing evaluation."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'workflow/scripts'))
from finalize_openalphaseq_pairing import TrainingMatcher, binding_label, finalize


class PairingCohortTests(unittest.TestCase):
    def test_complete_training_substrings_and_suffix_fallback(self):
        matcher = TrainingMatcher(['ABCD', 'BCDXY', 'CDXYQ'])
        self.assertTrue(matcher.contains('ZZABCDR'))
        self.assertTrue(matcher.contains('ABCDXYQ'))
        self.assertTrue(matcher.contains('ZBCDXY'))
        self.assertFalse(matcher.contains('ZZABCZ'))
        self.assertFalse(matcher.contains('CDXY'))

    def test_any_reliable_positive_wins_conflicting_annotations(self):
        self.assertEqual(binding_label(1, 20), 'positive')
        self.assertEqual(binding_label(0, 1), 'negative')
        self.assertEqual(binding_label(0, 0), 'uncertain')

    def test_domain_overlap_novel_chain_masks_and_uncertain_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); training_root = root / 'training_numbering'; training_root.mkdir()
            old_h, old_l = 'A' * 70, 'C' * 70
            new_h, new_l = 'D' * 70, 'E' * 70
            training = root / 'training.csv'
            with training.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['split', 'resolved_H_seq', 'resolved_L_seq'])
                writer.writeheader(); writer.writerows([
                    dict(split='train', resolved_H_seq=old_h, resolved_L_seq=old_l),
                    dict(split='test', resolved_H_seq=new_h, resolved_L_seq=new_l)])
            schema = 'CREATE TABLE domains(component_id TEXT,sequence TEXT,status TEXT,chain TEXT,target TEXT,cdr3_indices TEXT)'
            with sqlite3.connect(training_root / 'domains.sqlite') as db:
                db.execute(schema)
                db.executemany('INSERT INTO domains VALUES (?,?,?,?,?,?)', [
                    ('train_h', old_h, 'valid', 'heavy', old_h, '[65]'),
                    ('train_l', old_l, 'valid', 'light', old_l, '[65]')])
            with sqlite3.connect(root / 'domains.sqlite') as db:
                db.execute(schema)
                db.executemany('INSERT INTO domains VALUES (?,?,?,?,?,?)', [
                    ('h', new_h, 'valid', 'heavy', new_h, '[65]'),
                    ('h_overhang', 'G' + new_h, 'valid', 'heavy', new_h, '[65]'),
                    ('old_h', old_h, 'valid', 'heavy', old_h, '[65]'),
                    ('old_l', old_l + 'R', 'valid', 'light', old_l, '[65]'),
                    ('l', new_l, 'valid', 'light', new_l, '[65]')])
            with sqlite3.connect(root / 'raw.sqlite') as db:
                db.executescript('''
                CREATE TABLE antibodies(id TEXT,components TEXT,kind TEXT,families TEXT);
                CREATE TABLE observations(antibody_id TEXT,antigen_id TEXT,experiment TEXT,
                                          affinity REAL,above_background INTEGER,sufficient_replicates INTEGER);
                ''')
                db.executemany('INSERT INTO antibodies VALUES (?,?,?,?)', [
                    ('her2', '["h","old_l"]', 'scfv', '["HER2"]'),
                    ('her2_duplicate', '["h_overhang","old_l"]', 'scfv', '["HER2"]'),
                    ('old', '["old_h","old_l"]', 'scfv', '["HER2"]'),
                    ('tigit', '["h","l"]', 'scfv', '["TIGIT"]'),
                    ('ace2', '[]', 'full_sequence_fallback', '["RBD"]')])
                db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?)', [
                    ('her2', 'a', 'x', 1., 0, 1), ('her2_duplicate', 'a', 'x', 2., 1, 1),
                    ('tigit', 'a', 'x', None, 0, 1), ('tigit', 'b', 'x', 1., 1, 0),
                    ('tigit', 'c', 'x', 1., 0, 1)])
            (root / 'raw_manifest.json').write_text('{}')
            for marker in (root / 'numbering_complete.json', training_root / 'numbering_complete.json'):
                marker.write_text(json.dumps(dict(complete=True, source_sha256='fixture')))
            with contextlib.redirect_stdout(io.StringIO()):
                audit = finalize(dict(experiment_root=str(root), training_records=str(training)))
            self.assertEqual(audit['unique_queries'], 2)
            self.assertEqual(audit['retained_raw_constructs'], 3)
            self.assertEqual(audit['strict_unseen_queries'], 1)
            self.assertEqual(audit['heavy_novel_queries'], 2)
            self.assertEqual(audit['unique_novel_targets'], 2)
            self.assertEqual(audit['conflicting_pairs'], 1)
            self.assertEqual(audit['labels'], dict(positive=1, negative=1, uncertain=2))
            self.assertEqual(audit['exclusions']['all_domains_seen_in_training'], 1)
            with sqlite3.connect(root / 'cohort.sqlite') as db:
                self.assertEqual(db.execute('SELECT light_novel,strict_unseen,raw_constructs FROM queries WHERE families=?',
                                            ('["HER2"]',)).fetchone(), (0, 0, 2))


if __name__ == '__main__':
    unittest.main()
