"""Exercise Delta snapshot selection and interruption/corruption recovery offline."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "workflow/scripts/download_asd.py"
spec = importlib.util.spec_from_file_location("download_asd", SCRIPT)
asd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asd)


class FakeDrive:
    def __init__(self):
        schema = {"fields": [{"name": "chains", "type": {
            "type": "struct", "fields": [
                {"name": "antibody_chains"}, {"name": "target_chains"},
            ]}}]}
        self.parts = {"old.parquet": b"PAR1obsoletePAR1", "new.parquet": b"PAR1current!PAR1"}
        self.logs = {
            "_delta_log/00000000000000000000.json": [
                {"protocol": {"minReaderVersion": 1}},
                {"metaData": {"id": "test-table", "format": {"provider": "parquet"},
                              "schemaString": json.dumps(schema)}},
                self.add("old.parquet", 7),
            ],
            "_delta_log/00000000000000000001.json": [
                {"remove": {"path": "old.parquet"}}, self.add("new.parquet", 5),
            ],
        }
        self.calls = []
        self.fail = None

    def add(self, path, rows):
        return {"add": {"path": path, "size": len(self.parts[path]),
                        "stats": json.dumps({"numRecords": rows})}}

    def download_folder(self, **kwargs):
        return [SimpleNamespace(id=name, path=name) for name in [*self.logs, *self.parts]]

    def download(self, id, output, **kwargs):
        self.calls.append(id)
        if id == self.fail:
            raise RuntimeError("simulated interrupted transfer")
        data = ("\n".join(json.dumps(action) for action in self.logs[id]).encode()
                if id in self.logs else self.parts[id])
        Path(output).write_bytes(data)
        return output


class ASDDownloadTests(unittest.TestCase):
    def run_download(self, root, drive, **kwargs):
        with redirect_stdout(io.StringIO()):
            return asd.download(output=root, quiet=True, gdown_module=drive, **kwargs)

    def test_download_only_active_parts_and_resume_without_transfers(self):
        with TemporaryDirectory() as root:
            drive = FakeDrive()
            manifest = self.run_download(root, drive)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["rows_from_delta_statistics"], 5)
            self.assertEqual(manifest["data_file_count"], 1)
            self.assertNotIn("old.parquet", drive.calls)
            self.assertEqual((Path(root) / "new.parquet").read_bytes(), drive.parts["new.parquet"])
            drive.calls.clear()
            self.run_download(root, drive)
            self.assertEqual(drive.calls, [])

    def test_metadata_only_never_marks_data_complete(self):
        with TemporaryDirectory() as root:
            drive = FakeDrive()
            manifest = self.run_download(root, drive, metadata_only=True)
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"], "metadata_only")
            self.assertFalse((Path(root) / "new.parquet").exists())
            self.assertTrue(self.run_download(root, drive)["complete"])

    def test_corruption_is_repaired_even_when_file_size_and_magic_match(self):
        with TemporaryDirectory() as root:
            drive = FakeDrive()
            self.run_download(root, drive)
            (Path(root) / "new.parquet").write_bytes(b"PAR1corrupt!PAR1")
            drive.calls.clear()
            self.run_download(root, drive)
            self.assertEqual(drive.calls, ["new.parquet"])
            self.assertEqual((Path(root) / "new.parquet").read_bytes(), drive.parts["new.parquet"])

    def test_interrupted_run_records_failure_and_reuses_finished_files(self):
        with TemporaryDirectory() as root:
            drive = FakeDrive()
            drive.fail = "new.parquet"
            with self.assertRaisesRegex(RuntimeError, "interrupted transfer"):
                self.run_download(root, drive)
            manifest = json.loads((Path(root) / asd.MANIFEST).read_text())
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"], "failed")
            drive.calls.clear()
            drive.fail = None
            self.assertTrue(self.run_download(root, drive)["complete"])
            self.assertEqual(drive.calls, ["new.parquet"])

    def test_truncated_and_html_responses_are_not_published(self):
        for payload in [b"PAR1cut", b"<html>not a parquet table</html>"]:
            with self.subTest(payload=payload), TemporaryDirectory() as root:
                drive = FakeDrive()
                drive.parts["new.parquet"] = payload
                with self.assertRaises(ValueError):
                    self.run_download(root, drive)
                self.assertFalse((Path(root) / "new.parquet").exists())
                self.assertFalse((Path(root) / "new.parquet.download").exists())
                manifest = json.loads((Path(root) / asd.MANIFEST).read_text())
                self.assertFalse(manifest["complete"])

    def test_metadata_failure_preserves_previous_data_checksums(self):
        with TemporaryDirectory() as root:
            drive = FakeDrive()
            complete = self.run_download(root, drive)
            first_log = "_delta_log/00000000000000000000.json"
            (Path(root) / first_log).write_text("invalid JSON")
            drive.fail = first_log
            with self.assertRaises(RuntimeError):
                self.run_download(root, drive)
            failed = json.loads((Path(root) / asd.MANIFEST).read_text())
            self.assertFalse(failed["complete"])
            self.assertEqual(failed["files"]["new.parquet"]["sha256"],
                             complete["files"]["new.parquet"]["sha256"])

    def test_missing_history_or_active_part_fails(self):
        for missing in ["_delta_log/00000000000000000000.json", "new.parquet"]:
            with self.subTest(missing=missing), TemporaryDirectory() as root:
                drive = FakeDrive()
                if missing in drive.logs:
                    del drive.logs[missing]
                else:
                    del drive.parts[missing]
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.run_download(root, drive)

    def test_existing_manifest_pins_snapshot_when_upstream_adds_version(self):
        with TemporaryDirectory() as root:
            drive = FakeDrive()
            self.run_download(root, drive)
            later_log = "_delta_log/00000000000000000002.json"
            drive.logs[later_log] = [{"remove": {"path": "new.parquet"}}]
            drive.calls.clear()
            manifest = self.run_download(root, drive)
            self.assertEqual(manifest["delta_version"], 1)
            self.assertNotIn(later_log, drive.calls)
            self.assertTrue(manifest["complete"])

    def test_paths_cannot_escape_output(self):
        for name in ["../outside.parquet", "/tmp/outside.parquet", "%2e%2e/outside", "C:\\outside"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                asd.safe_path(name)


if __name__ == "__main__":
    unittest.main()
