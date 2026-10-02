import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from time_command import run
from vram_monitor import memory_for_pids, descendants
from report_gpu_results import _timing_summary


class WallTimeTests(unittest.TestCase):
    def test_gpu_memory_excludes_other_jobs(self):
        self.assertEqual(memory_for_pids('100, 500\n101, 250\n999, 9000\n', {100, 101}), 750)
        self.assertEqual(memory_for_pids('', {100}), 0)
        import os
        self.assertIn(os.getpid(), descendants(os.getpid()))


    def test_success_and_failure_are_retained_and_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'times.jsonl'
            self.assertEqual(run([sys.executable, '-c', 'pass'], path, 'esm2', 'weights'), 0)
            self.assertEqual(run([sys.executable, '-c', 'raise SystemExit(7)'], path, 'esm2', 'inference'), 7)
            timing, error = _timing_summary(path, 'esm2')
            self.assertIsNone(error)
            self.assertEqual(len(timing['attempts']), 2)
            self.assertEqual(timing['attempts'][1]['exit_code'], 7)
            self.assertGreater(timing['wall_seconds'], 0)
            self.assertEqual(timing['wall_seconds'], sum(timing['stage_wall_seconds'].values()))
            self.assertEqual(_timing_summary(path, 'esm3'), (None, None))

    def test_missing_or_corrupt_timing_does_not_break_report(self):
        self.assertEqual(_timing_summary(None, 'esm2'), (None, None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'times.jsonl'
            path.write_text('{')
            timing, error = _timing_summary(path, 'esm2')
            self.assertIsNone(timing)
            self.assertIsNotNone(error)

    def test_repeated_weight_attempts_count_toward_same_embedder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'times.jsonl'
            path.write_text('\n'.join(json.dumps({'embedder': tag, 'stage': 'weights', 'wall_seconds': seconds}) for tag, seconds in [('esm2', 2), ('esm3', 100), ('esm2', 3)]))
            timing, _ = _timing_summary(path, 'esm2')
            self.assertEqual(timing['wall_seconds'], 5)
