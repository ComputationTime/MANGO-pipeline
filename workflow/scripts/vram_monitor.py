"""Sample NVIDIA memory for a command's process tree, excluding other jobs."""
from pathlib import Path
import subprocess
import threading


def descendants(root):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            # comm may contain spaces or parentheses; fields after it start at state.
            fields = path.read_text().rsplit(')', 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {root}
    while True:
        extra = {pid for pid, parent in parents.items() if parent in found} - found
        if not extra:
            return found
        found.update(extra)


def memory_for_pids(output, pids):
    total = 0.0
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, memory = (value.strip() for value in line.split(',', 1))
        if int(pid) in pids:
            total += float(memory)
    return total


class VramMonitor:
    interval = 0.5

    def __init__(self, pid):
        self.pid = pid
        self.peak = None
        self.samples = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                pids = descendants(self.pid)
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                     '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                used = memory_for_pids(output, pids)
                self.peak = max(self.peak or 0.0, used)
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.error = f'{type(exc).__name__}: {exc}'
            self.stop.wait(self.interval)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)
        return {'peak_vram_mib': self.peak, 'vram_samples': self.samples,
                'vram_sample_interval_seconds': self.interval,
                'vram_measurement_error': self.error,
                'vram_scope': 'Sampled peak sum of NVIDIA compute-process memory for command descendants. Excludes desktop and independent jobs; short peaks can be missed. Not a proven minimum VRAM requirement.'}
