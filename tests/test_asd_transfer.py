"""Binding policy, target deduplication and compact ASD embedding contracts."""
import csv
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow' / 'scripts'))
from asd_transfer_embeddings import embed_controls, finish, write_index
from chain_examples import example_chains
from embed_antigen_control import control_matrix
import evaluate_mango
import model_common as mc
from prepare_asd_transfer import candidate_query, pdb_ids, select_examples
from train_mango import _rows


class ASDTransferTests(unittest.TestCase):
    def rows(self):
        rows = []
        for split in ('train', 'val', 'test'):
            for i in range(2):
                rows.append(dict(id=f'{split}{i}', split=split, resolved_ag_seq='ACD' if i == 0 else 'EF',
                    antigen_chains='AG', resolved_H_seq='GHI', resolved_L_seq='KLM',
                    target_contract='imgt_variable_vh_vl_v1', ab_ag_cluster=split))
        return rows

    def write_rows(self, path, rows):
        with Path(path).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_each_antigen_chain_target_is_scored_once(self):
        rows = [dict(id='a', split='train', resolved_ag_seq='ACD', resolved_H_seq='GHI', resolved_L_seq='KLM'),
                dict(id='b', split='train', resolved_ag_seq='ACD', resolved_H_seq='GHV', resolved_L_seq='KLM'),
                dict(id='c', split='train', resolved_ag_seq='ACD', resolved_H_seq='GHI', resolved_L_seq='KLV'),
                dict(id='d', split='train', resolved_ag_seq='ACD', resolved_H_seq='GHV', resolved_L_seq='KLV')]
        selected = select_examples(rows)
        self.assertEqual([r['example_target_chains'] for r in selected], ['heavy;light', 'heavy', 'light'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.csv'
            self.write_rows(path, selected)
            examples = _rows(path, ['train'], ['heavy', 'light'])
            self.assertEqual([(r.id, r.chain) for r in examples],
                             [('a', 'heavy'), ('a', 'light'), ('b', 'heavy'), ('c', 'light')])
            evaluated = evaluate_mango._rows(path, 'train')
            self.assertEqual([example_chains(r) for r in evaluated], [('heavy', 'light'), ('heavy',), ('light',)])
        wrong = [dict(rows[0]), dict(rows[0], id='wrong', split='test')]
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            select_examples(wrong)

    def test_target_mask_is_pinned_and_legacy_fingerprints_stay_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.csv'
            rows = self.rows()
            self.write_rows(path, rows)
            legacy = mc.comparison_cohort(path)
            for row in rows:
                row['example_target_chains'] = 'heavy;light'
            self.write_rows(path, rows)
            masked = mc.comparison_cohort(path)
            self.assertNotEqual(legacy, masked)
            rows[0]['example_target_chains'] = 'heavy'
            self.write_rows(path, rows)
            self.assertNotEqual(masked, mc.comparison_cohort(path))
            rows[0]['example_target_chains'] = ''
            self.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'example_target_chains'):
                mc.comparison_cohort(path)

    def test_native_pdb_normalization(self):
        self.assertEqual(pdb_ids('pdb_00001j1p,2XYZ'), {'1j1p', '2xyz'})

    def test_any_positive_policy_keeps_affinity_disagreements_and_all_buzz_bins(self):
        with sqlite3.connect(':memory:') as db:
            db.executescript('''
                CREATE TABLE rows(row_id INTEGER,source TEXT,antigen_ids TEXT,affinity TEXT);
                CREATE TABLE targets(row_id INTEGER,heavy TEXT,light TEXT,target_pdb TEXT);
                CREATE TABLE eligibility(row_id INTEGER,family_safe INTEGER,homology_safe INTEGER,
                    pdb_safe INTEGER,heavy_safe INTEGER,light_safe INTEGER,binding_positive INTEGER);
                CREATE TABLE antigens(antigen_id TEXT,cath_families TEXT);
                INSERT INTO antigens VALUES ('ag','1.2.3.4');
            ''')
            cases = [('other', '6.0', 0, 'shared'), ('other', '8.0', 1, 'shared'),
                     ('other', 'false', 0, 'negative_only'), ('buzz', 'h', 1, 'h'),
                     ('buzz', 'm', 0, 'm'), ('buzz', 'l', 0, 'l'),
                     ('other', 'ddg', 0, 'relative_only')]
            for i, (source, affinity, positive, target) in enumerate(cases):
                db.execute('INSERT INTO rows VALUES (?,?,?,?)', (i, source, 'ag', affinity))
                db.execute('INSERT INTO targets VALUES (?,?,?,?)', (i, target, 'L', ''))
                db.execute('INSERT INTO eligibility VALUES (?,1,1,1,1,1,?)', (i, positive))
            with_buzz = list(db.execute(candidate_query()))
            self.assertEqual({row[1] for row in with_buzz}, {'shared', 'h', 'm', 'l'})
            self.assertEqual(len([r for r in with_buzz if r[1] == 'shared']), 1)
            self.assertEqual({r[1] for r in db.execute(candidate_query(False))}, {'shared'})
            db.execute("UPDATE eligibility SET family_safe=0 WHERE row_id=4")
            self.assertNotIn('m', {r[1] for r in db.execute(candidate_query())})

    def test_compact_controls_reproduce_tensors_and_keep_shuffle_within_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / 'records.csv'
            rows = self.rows()
            self.write_rows(records, rows)
            source = root / 'asd' / 'embeddings' / 'antigen' / 'esm2'
            cache = root / 'cache'
            cache.mkdir()
            matrices = {'ag0': torch.arange(3 * 1280, dtype=torch.float32).reshape(3, 1280),
                        'ag1': torch.ones(2, 1280)}
            for key, matrix in matrices.items():
                torch.save(dict(embedding=matrix, shape=list(matrix.shape)), cache / f'{key}.pt')
            index = source / 'index.sqlite'
            assignments = [(r['split'], r['id'], f'ag{i % 2}', 3 if i % 2 == 0 else 2)
                           for i, r in enumerate(rows)]
            write_index(index, assignments)
            finish(source, source / 'complete.json', dict(mode='esm2', width=1280,
                   index=str(index.resolve()), cache_dir=str(cache.resolve())), records, len(rows))
            baseline = root / 'asd' / 'runs' / 'esm2' / 'eval.json'
            baseline.parent.mkdir(parents=True)
            baseline.write_text(json.dumps(dict(embedder='esm2', cohort_sha256=mc.comparison_cohort(records))))
            study = dict(experiment_root=str(root))
            tags = ['control_constant', 'control_random', 'control_constant_l1', 'control_random_l1',
                    'control_pooled_esm2', 'control_shuffled']
            for tag in tags:
                directory = source.parent / tag
                embed_controls(study, 'asd', tag, str(records), str(directory / 'complete.json'))
                self.assertFalse((directory / 'train' / 'train0.pt').exists())
                result = mc.load_embedding(directory / 'train' / 'train0.pt')
                if tag == 'control_pooled_esm2':
                    self.assertTrue(torch.equal(result[0], matrices['ag0'].mean(0, keepdim=True)))
                elif tag == 'control_shuffled':
                    self.assertTrue(torch.equal(result[0], matrices['ag1']))
                    with sqlite3.connect(directory / 'index.sqlite') as db:
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM records').fetchone()[0], 6)
                    pairing = json.loads((directory / 'biological_pairing_audit.json').read_text())
                    self.assertEqual(pairing['train']['same_antigen_sequence'], 0)
                else:
                    fixed = tag.endswith('_l1')
                    expected = control_matrix('train0', dict(mode='constant' if 'constant' in tag else 'random',
                        width=1280, seed=0, shape_mode='fixed' if fixed else 'match_source', tokens=1), shape=(3, 1280))
                    self.assertTrue(torch.equal(result[0], expected))
            with self.assertRaisesRegex(FileNotFoundError, 'absent'):
                mc.load_embedding(source / 'train' / 'unknown.pt')
            # Evaluation must honor the mask before attempting an embedding load.
            row = SimpleNamespace(id='train0', split='train', resolved_H_seq='ACD', resolved_L_seq='EF',
                                  example_target_chains='heavy')
            model = SimpleNamespace(target_token_ids=lambda seq: torch.zeros(1, len(seq)+2, dtype=torch.long),
                n_target_tokens=lambda ids: ids.shape[1]-1, loss=lambda antigen, chain, ids: torch.tensor(2.0))
            result = evaluate_mango._split_nll(model, [row], root / 'asd' / 'embeddings', 'esm2', 'heavy', 'cpu')
            self.assertEqual(result, (2.0, 4, 1, 0))
            result = evaluate_mango._split_nll(model, [row], '/missing', 'esm2', 'light', 'cpu')
            self.assertEqual(result[1:], (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
