import csv
import json
from pathlib import Path
import random
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workflow/scripts'))
import embed_batch
from embed_antigen_one_hot import VOCAB, encode_one_hot, one_hot
from embedding_benchmark import EmbeddingBenchmark, local_assets_only, serialization
from report_gpu_results import _embedding_benchmark, report


class OneHotVectorizationTests(unittest.TestCase):
    def test_exact_reference_including_unknowns_and_unicode(self):
        rng = random.Random(13)
        sequences = ['', VOCAB, 'Xx?*-é𝒜\ud800|', ''.join(rng.choices(VOCAB + 'XYZ?éa', k=1000))]
        for seq in sequences:
            reference = torch.zeros((len(seq), len(VOCAB)), dtype=torch.float32)
            for i, aa in enumerate(seq):
                j = VOCAB.find(aa)
                if j >= 0:
                    reference[i, j] = 1
            actual = encode_one_hot(seq)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, reference))

    def test_multiple_chains_and_serialized_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'embedding.pt'
            one_hot('', 'x', str(path), 'resolved', 'one_hot',
                    row={'id': 'x', 'antigen_chains': 'A,B,C', 'resolved_ag_seq': 'AC,,Xé'})
            saved = torch.load(path, weights_only=False)
            self.assertTrue(torch.equal(saved['embedding'], encode_one_hot('AC||Xé')))
            self.assertEqual(saved['shape'], [6, 21])
            self.assertEqual(saved['chains'], ['A', 'B', 'C'])
            self.assertTrue(saved['chain_separator_token'])


class EmbeddingTimingTests(unittest.TestCase):
    def rows(self):
        return [{'id': 'a', 'split': 'train', 'antigen_chains': 'A,B', 'resolved_ag_seq': 'AC,X'},
                {'id': 'b', 'split': 'test', 'antigen_chains': 'A', 'resolved_ag_seq': 'DEF'}]

    def test_synchronized_phase_accounting_and_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = EmbeddingBenchmark(self.rows()[:1], 'one_hot', 'one_hot', {}, 'resolved', tmp)
            # Deterministic timestamps: load=2, record total=6, serialization=2.
            with patch('embedding_benchmark.time.perf_counter', side_effect=[0, 2, 3, 4, 6, 9]), patch.object(benchmark, 'sync') as sync:
                with benchmark.loading():
                    pass
                with benchmark.record(self.rows()[0]):
                    with serialization():
                        pass
            self.assertEqual(sync.call_count, 5)
            result = benchmark.finish()
            self.assertEqual(result['model_load_seconds'], 2)
            self.assertEqual(result['compute_seconds'], 4)
            self.assertEqual(result['serialization_seconds'], 2)
            self.assertEqual(result['total_embedding_seconds'], 8)
            self.assertEqual(result['seconds_per_structure'], 8)
            self.assertEqual(result['fresh_residues'], 3)  # no separator
            self.assertEqual(result['residues_per_second'], 0.75)
            self.assertTrue(result['fresh_comparison_eligible'])
            benchmark.cuda = True
            with patch('torch.cuda.synchronize') as cuda_sync:
                benchmark.sync()
                cuda_sync.assert_called_once()

    def test_fresh_cached_and_partial_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / 'records.csv'
            with records.open('w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=self.rows()[0].keys())
                writer.writeheader()
                writer.writerows(self.rows())
            out = root / 'one_hot'
            kwargs = dict(records_csv=str(records), output_dir=str(out), marker=str(out / '.batch_complete.json'),
                          kind='antigen', tag='one_hot', method='one_hot', spec={'method': 'one_hot'},
                          seq_source='resolved', splits=['train', 'test'],
                          implementation=str(ROOT / 'workflow/scripts/embed_antigen_one_hot.py'))
            embed_batch.batch_embed(**kwargs)
            first = json.loads((out / 'embedding_benchmark.json').read_text())
            self.assertEqual(first['fresh_records'], 2)
            self.assertTrue(first['fresh_comparison_eligible'])
            with patch('embed_batch._runtime', side_effect=AssertionError('Cached runs must not load a model')):
                embed_batch.batch_embed(**kwargs)
            cached = json.loads((out / 'embedding_benchmark.json').read_text())
            self.assertEqual(cached['reused_records'], 2)
            self.assertIsNone(cached['total_embedding_seconds'])
            self.assertIsNone(cached['residues_per_second'])
            self.assertFalse(cached['fresh_comparison_eligible'])
            (out / 'test/b.pt').unlink()
            embed_batch.batch_embed(**kwargs)
            partial = json.loads((out / 'embedding_benchmark.json').read_text())
            self.assertEqual(partial['fresh_records'], 1)
            self.assertEqual(partial['reused_records'], 1)
            self.assertEqual(partial['fresh_residues'], 3)
            self.assertFalse(partial['fresh_comparison_eligible'])
            self.assertEqual(len(list((out / 'benchmarks').glob('*.json'))), 3)
            embed_batch.batch_embed(**kwargs, force_fresh=True)
            forced = json.loads((out / 'embedding_benchmark.json').read_text())
            self.assertEqual(forced['fresh_records'], 2)
            self.assertTrue(forced['fresh_comparison_eligible'])

    def test_failure_is_archived_without_valid_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = EmbeddingBenchmark(self.rows(), 'one_hot', 'one_hot', {}, 'resolved', tmp)
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with benchmark.record(self.rows()[0]):
                    raise RuntimeError('intentional')
            result = benchmark.finish('intentional')
            self.assertEqual(result['failed_records'], 1)
            self.assertEqual(result['unattempted_records'], 1)
            self.assertFalse(result['fresh_comparison_eligible'])
            self.assertIsNone(result['total_embedding_seconds'])

    def test_gpu_peaks_and_boundary_synchronization(self):
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available', return_value=True), patch('embedding_benchmark.VramMonitor') as monitor, patch('torch.cuda.synchronize') as sync, patch('torch.cuda.reset_peak_memory_stats') as reset, patch('torch.cuda.max_memory_allocated', return_value=2**20), patch('torch.cuda.max_memory_reserved', return_value=2 * 2**20), patch('torch.cuda.get_device_name', return_value='test GPU'):
            monitor.return_value.finish.return_value = {'peak_vram_mib': 3}
            benchmark = EmbeddingBenchmark(self.rows()[:1], 'esm2', 'esm2', {}, 'resolved', tmp)
            with benchmark.loading():
                pass
            with benchmark.record(self.rows()[0]):
                with serialization():
                    pass
            result = benchmark.finish()
            reset.assert_called_once()
            self.assertEqual(sync.call_count, 6)
            self.assertEqual(result['peak_vram_mib'], 3)
            self.assertEqual(result['peak_torch_allocated_mib'], 1)
            self.assertEqual(result['peak_torch_reserved_mib'], 2)

    def test_report_keeps_implementation_groups_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timing = root / 'launch.jsonl'
            timing.touch()
            specs = []
            for tag, suite in [('a', 'v1'), ('b', 'v2')]:
                benchmark = root / f'{tag}.json'
                benchmark.write_text(json.dumps({'schema_version': 1, 'tag': tag,
                    'status': 'complete', 'launcher_invocation_id': 'launch', 'fresh_comparison_eligible': True,
                    'suite_id': suite, 'cohort_id': 'same', 'fresh_records': 1,
                    'reused_records': 0, 'total_embedding_seconds': 4}))
                specs.append(dict(embedder=tag, run_id=tag, embedding=str(root/'missing'),
                    checkpoint=str(root/'missing'), evaluation=str(root/'missing'),
                    predictions=str(root/'missing'), timings=str(timing),
                    embedding_benchmark=str(benchmark)))
            report(specs, str(root/'preflight'), [], [], str(root/'report.json'), str(root/'report.csv'))
            result = json.loads((root/'report.json').read_text())
            self.assertEqual([group['embedders'] for group in result['embedding_comparison_groups']], [['a'], ['b']])
            with (root/'report.csv').open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]['embedding_total_seconds'], '4')
            self.assertEqual(rows[0]['launcher_timing_scope'], 'end_to_end_commands_including_setup')

    def test_start_replaces_previous_completed_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'embedding_benchmark.json'
            path.write_text(json.dumps({'fresh_comparison_eligible': True}))
            benchmark = EmbeddingBenchmark(self.rows(), 'one_hot', 'one_hot', {}, 'resolved', tmp)
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'running')
            self.assertFalse(result['fresh_comparison_eligible'])
            self.assertEqual(result['invocation_id'], benchmark.invocation_id)

    def test_no_downloads_in_measured_work(self):
        with local_assets_only(), socket.socket() as sock:
            with self.assertRaisesRegex(RuntimeError, 'weights target'):
                sock.connect(('127.0.0.1', 9))

    def test_implementation_change_invalidates_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = EmbeddingBenchmark(self.rows(), 'one_hot', 'one_hot', {}, 'resolved', tmp)
            with patch('embedding_benchmark.suite_fingerprint', return_value='changed'):
                result = benchmark.finish()
            self.assertFalse(result['implementation_unchanged'])
            self.assertFalse(result['fresh_comparison_eligible'])

    def test_old_launcher_measurement_is_not_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timing = root / 'new-launch.jsonl'
            timing.touch()
            path = root / 'benchmark.json'
            path.write_text(json.dumps({'schema_version': 1, 'status': 'complete', 'suite_id': 'v1', 'cohort_id': 'c', 'launcher_invocation_id': 'old-launch', 'fresh_comparison_eligible': True}))
            result, error = _embedding_benchmark({'embedding_benchmark': str(path), 'timings': str(timing)})
            self.assertIsNone(error)
            self.assertFalse(result['current_invocation'])
            self.assertFalse(result['comparison_eligible'])
