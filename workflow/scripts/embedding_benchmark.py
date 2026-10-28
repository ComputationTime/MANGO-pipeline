"""Fresh-only embedding timings. Assets must already be available locally."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch
import uuid

import torch
from vram_monitor import VramMonitor

_ACTIVE = ContextVar('embedding_benchmark', default=None)
GPU_METHODS = {'esm2', 'esm3', 'esmif', 'proteinmpnn'}


def suite_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = set()
    for pattern in ['workflow/scripts/embed*.py', 'workflow/scripts/vram_monitor.py',
                    'workflow/scripts/device_common.py', 'workflow/scripts/lib/*.py',
                    'workflow/envs/embed*', 'mango/utils/MPNN*.py']:
        paths.update(root.glob(pattern))
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def local_assets_only():
    """Never include a hidden model download in measured loading/computation."""
    def deny(*args, **kwargs):
        raise RuntimeError('Embedding benchmark requires local assets; run the weights target first.')
    with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}):
        with patch.object(socket.socket, 'connect', deny), patch.object(socket.socket, 'connect_ex', deny):
            yield


@contextmanager
def serialization():
    benchmark = _ACTIVE.get()
    if benchmark is None:
        yield
        return
    benchmark.sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        benchmark.sync()
        benchmark.serialization_seconds += time.perf_counter() - start


class EmbeddingBenchmark:
    def __init__(self, rows, method, tag, spec, seq_source, output_dir):
        self.rows = rows
        self.method = method
        self.tag = tag
        self.spec = spec
        self.seq_source = seq_source
        self.output_dir = Path(output_dir)
        self.suite_id = suite_fingerprint()
        self.cohort_id = hashlib.sha256(json.dumps({'rows': rows, 'residue_source': 'resolved' if seq_source == 'structure' else seq_source}, sort_keys=True).encode()).hexdigest()
        self.invocation_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.launch_id = os.environ.get('MANGO_LAUNCH_ID')
        self.cuda = method in GPU_METHODS and torch.cuda.is_available()
        self.monitor = None
        self.started = False
        self.model_load_seconds = 0.0
        self.serialization_seconds = 0.0
        self.records = []
        # Replace a previous result before touching outputs, so a killed retry
        # cannot leave the previous successful benchmark looking current.
        self.publish({'schema_version': 1, 'scope': 'embedding_only', 'tag': tag,
                      'status': 'running', 'suite_id': self.suite_id,
                      'cohort_id': self.cohort_id, 'invocation_id': self.invocation_id,
                      'launcher_invocation_id': self.launch_id,
                      'fresh_comparison_eligible': False})

    def publish(self, result, archive=False):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(result, indent=2, sort_keys=True) + '\n'
        if archive:
            path = self.output_dir / 'benchmarks' / f'{self.invocation_id}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(contents)
            print(f'[{self.tag}] embedding benchmark -> {path}', flush=True)
        temporary = self.output_dir / f'.benchmark-{self.invocation_id}.tmp'
        temporary.write_text(contents)
        temporary.replace(self.output_dir / 'embedding_benchmark.json')

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def loading(self):
        if self.cuda:
            self.sync()
            torch.cuda.reset_peak_memory_stats()
            self.monitor = VramMonitor(os.getpid())
            self.monitor.start()
        self.started = True
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.model_load_seconds += time.perf_counter() - start

    def residues(self, row):
        column = 'expected_ag_seq' if self.seq_source == 'expected' else 'resolved_ag_seq'
        # Input biological residues, including unknowns; never separator tokens.
        return sum(len(sequence) for sequence in row[column].split(','))

    def reused(self, row):
        self.records.append({'id': row['id'], 'split': row['split'], 'status': 'reused',
                             'residues': self.residues(row), 'compute_seconds': None,
                             'serialization_seconds': None})

    @contextmanager
    def record(self, row):
        self.sync()
        before_serialization = self.serialization_seconds
        start = time.perf_counter()
        token = _ACTIVE.set(self)
        status = 'fresh'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            self.sync()
            elapsed = time.perf_counter() - start
            _ACTIVE.reset(token)
            serialization_time = self.serialization_seconds - before_serialization
            self.records.append({'id': row['id'], 'split': row['split'], 'status': status,
                                 'residues': self.residues(row),
                                 'compute_seconds': max(0.0, elapsed - serialization_time),
                                 'serialization_seconds': serialization_time})

    def finish(self, error=None):
        memory = self.monitor.finish() if self.monitor else {}
        fresh = [row for row in self.records if row['status'] == 'fresh']
        reused = sum(row['status'] == 'reused' for row in self.records)
        compute = sum(row['compute_seconds'] for row in fresh)
        serialize = sum(row['serialization_seconds'] for row in fresh)
        residue_count = sum(row['residues'] for row in fresh)
        total = self.model_load_seconds + compute + serialize
        unchanged = self.suite_id == suite_fingerprint()
        eligible = error is None and unchanged and len(fresh) == len(self.rows)
        result = {
            'schema_version': 1, 'scope': 'embedding_only', 'tag': self.tag,
            'method': self.method, 'spec': self.spec, 'seq_source': self.seq_source,
            'suite_id': self.suite_id, 'cohort_id': self.cohort_id,
            'implementation_unchanged': unchanged, 'invocation_id': self.invocation_id,
            'launcher_invocation_id': self.launch_id,
            'started_at': self.started_at, 'finished_at': datetime.now(timezone.utc).isoformat(),
            'status': 'failed' if error else 'complete', 'error': error,
            'fresh_comparison_eligible': eligible,
            'comparison_exclusion': ('failed' if error else 'implementation_changed' if not unchanged else 'reused_records' if reused else 'incomplete' if not eligible else None),
            'selected_records': len(self.rows), 'fresh_records': len(fresh),
            'reused_records': reused,
            'failed_records': sum(row['status'] == 'failed' for row in self.records),
            'unattempted_records': len(self.rows) - len(self.records),
            'fresh_residues': residue_count,
            'model_load_seconds': self.model_load_seconds if self.started else None,
            'compute_seconds': compute if fresh else None,
            'serialization_seconds': serialize if fresh else None,
            'total_embedding_seconds': total if fresh and error is None and unchanged else None,
            'seconds_per_structure': total / len(fresh) if fresh and error is None and unchanged else None,
            'residues_per_second': residue_count / compute if compute > 0 and error is None and unchanged else None,
            'peak_vram_mib': memory.get('peak_vram_mib') if self.cuda else (0.0 if fresh else None),
            'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if self.cuda and self.started else None,
            'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if self.cuda and self.started else None,
            'vram_sampling': memory,
            'device': (torch.cuda.get_device_name() if self.started else 'cuda (not initialized)') if self.cuda else 'cpu',
            'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
            'torch_cpu_threads': torch.get_num_threads(),
            'records': self.records,
        }
        self.publish(result, archive=True)
        return result
