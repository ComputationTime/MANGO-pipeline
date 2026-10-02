"""Measure a launcher command, retaining its exit code and a JSONL timing record."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from vram_monitor import VramMonitor


def run(command, output, embedder, stage):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    returncode = 1
    monitor = None
    try:
        process = subprocess.Popen(command)
        monitor = VramMonitor(process.pid)
        monitor.start()
        returncode = process.wait()
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        record = {
            'embedder': embedder, 'stage': stage,
            'started_at': started_at,
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': round(time.monotonic() - started, 6),
            'exit_code': returncode,
        }
        if monitor is not None:
            record.update(monitor.finish())
        # One append write keeps foreground/prefetch records from interleaving.
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (json.dumps(record) + '\n').encode())
        finally:
            os.close(fd)
        print(f"[{embedder}] {stage}: {record['wall_seconds']:.2f}s (exit {returncode})", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('a command is required')
    raise SystemExit(run(command, args.output, args.embedder, args.stage))
