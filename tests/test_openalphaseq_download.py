"""Offline checks for pinned snapshots, publisher checksums and safe reruns."""
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
import download_openalphaseq as downloader


class FakeHub:
    def __init__(self):
        self.payloads = {'README.md': b'Example dataset', 'LICENSE': b'Example license',
                         'data/YM_0005/data.parquet': b'PAR1first-table!PAR1',
                         'data/YM_0549/data.parquet': b'PAR1secondtablePAR1'}
        self.latest = downloader.DEFAULT_REVISION
        self.calls = []
        self.fail = None
        self.corrupt = None
        self.fail_once = False

    def info(self, revision):
        files = []
        for path, payload in self.payloads.items():
            item = dict(rfilename=path, size=len(payload),
                        blobId=hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest())
            if path.endswith('.parquet'):
                item['lfs'] = dict(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
            files.append(item)
        return dict(sha=self.latest if revision == 'main' else revision, siblings=files)

    def __call__(self, request, timeout):
        path = unquote(urlparse(request.full_url).path)
        if '/api/datasets/' in path:
            self.calls.append('metadata')
            return io.BytesIO(json.dumps(self.info(path.rsplit('/', 1)[-1])).encode())
        name = path.split('/resolve/', 1)[1].split('/', 1)[1]
        self.calls.append(name)
        if name == self.fail:
            if self.fail_once:
                self.fail = None
            raise URLError('simulated interrupted transfer')
        payload = self.payloads[name]
        if name == self.corrupt:
            payload = payload[:5] + b'X' + payload[6:]
        return io.BytesIO(payload)


class OpenAlphaSeqDownloadTests(unittest.TestCase):
    def run_download(self, root, hub, **kwargs):
        return downloader.download(output=root, opener=hub, quiet=True,
                                   retries=kwargs.pop('retries', 0), **kwargs)

    def test_full_download_checks_both_hash_types_and_rerun_reuses_every_file(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            result = self.run_download(root, hub)
            self.assertTrue(result['complete'])
            self.assertEqual(result['downloaded_files'], 4)
            self.assertEqual(result['experiments'], ['YM_0005', 'YM_0549'])
            self.assertTrue(all(v['verified'] for v in result['files'].values()))
            hub.calls.clear()
            reused = self.run_download(root, hub)
            self.assertEqual(hub.calls, ['metadata'])
            self.assertEqual(reused['downloaded_files'], 0)
            self.assertEqual(reused['reused_files'], 4)

    def test_same_size_parquet_corruption_is_repaired_by_publisher_hash(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            self.run_download(root, hub)
            name = 'data/YM_0005/data.parquet'
            p = Path(root) / name
            p.write_bytes(p.read_bytes()[:5] + b'X' + p.read_bytes()[6:])
            hub.calls.clear()
            self.run_download(root, hub)
            self.assertEqual(hub.calls, ['metadata', name])
            self.assertEqual(p.read_bytes(), hub.payloads[name])

    def test_non_lfs_metadata_corruption_is_repaired_by_git_blob_hash(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            self.run_download(root, hub)
            p = Path(root) / 'README.md'
            p.write_bytes(b'X' * p.stat().st_size)
            hub.calls.clear()
            self.run_download(root, hub)
            self.assertEqual(hub.calls, ['metadata', 'README.md'])

    def test_failure_is_recorded_and_finished_files_survive_resume(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            hub.fail = 'data/YM_0549/data.parquet'
            with self.assertRaises(URLError):
                self.run_download(root, hub)
            result = json.loads((Path(root) / downloader.MANIFEST).read_text())
            self.assertFalse(result['complete'])
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['files']['data/YM_0005/data.parquet']['verified'])
            hub.fail = None
            hub.calls.clear()
            resumed = self.run_download(root, hub)
            self.assertTrue(resumed['complete'])
            self.assertEqual(hub.calls, ['metadata', 'data/YM_0549/data.parquet'])

    def test_bad_transfer_is_never_published_and_transient_errors_retry(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            hub.corrupt = 'data/YM_0005/data.parquet'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.run_download(root, hub)
            self.assertFalse((Path(root) / hub.corrupt).exists())
            self.assertFalse((Path(root) / (hub.corrupt + '.download')).exists())
            hub.corrupt = None
            hub.fail, hub.fail_once = 'data/YM_0005/data.parquet', True
            with patch.object(downloader.time, 'sleep'):
                self.assertTrue(self.run_download(root, hub, retries=1)['complete'])

    def test_snapshot_stays_pinned_and_revision_change_requires_a_new_directory(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            self.run_download(root, hub)
            hub.latest = 'a' * 40
            self.assertEqual(self.run_download(root, hub)['revision'], downloader.DEFAULT_REVISION)
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                self.run_download(root, hub, revision='main')
            self.assertEqual(json.loads((Path(root) / downloader.MANIFEST).read_text())['revision'],
                             downloader.DEFAULT_REVISION)

    def test_metadata_plan_and_offline_verification(self):
        with TemporaryDirectory() as root:
            hub = FakeHub()
            plan = self.run_download(root, hub, metadata_only=True)
            self.assertFalse(plan['complete'])
            self.assertEqual(hub.calls, ['metadata'])
            self.run_download(root, hub)
            def no_network(*args, **kwargs):
                raise AssertionError('Offline verification attempted a network request')
            self.assertTrue(self.run_download(root, no_network, verify_only=True)['complete'])
            (Path(root) / 'README.md').write_text('broken')
            with self.assertRaises(ValueError):
                self.run_download(root, no_network, verify_only=True)

    def test_paths_symlinks_and_unmanaged_outputs_are_protected(self):
        for name in ('../outside', '/tmp/outside', '%2e%2e/outside', 'C:\\outside',
                     '.git/config', './README.md', downloader.MANIFEST, downloader.MANIFEST + '.tmp'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                downloader.safe_path(name)
        with TemporaryDirectory() as root, TemporaryDirectory() as outside:
            (Path(root) / 'data').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                downloader.local_path(Path(root).resolve(), 'data/outside')
        with TemporaryDirectory() as root:
            p = Path(root) / 'README.md'
            p.write_text('User-owned content')
            with self.assertRaisesRegex(ValueError, 'empty destination'):
                self.run_download(root, FakeHub())
            self.assertEqual(p.read_text(), 'User-owned content')


if __name__ == '__main__':
    unittest.main()
