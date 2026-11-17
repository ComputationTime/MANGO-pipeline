"""Notify the local desktop when an existing training launcher exits.

Linux pidfd waits are event-driven: no model calls, token usage or polling.
The watcher only reads training outputs and writes its own notification state.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def completion(root):
    path = root / 'results_report.json'
    report = json.loads(path.read_text()) if path.is_file() else {}
    runs = report.get('runs', [])
    final = [r for r in runs if r.get('phase') in {'sabdab2_scratch', 'sabdab2_finetuned'}]
    evaluated = sum(bool(r.get('complete')) for r in final)
    complete = report.get('complete') is True and len(final) == 8 and evaluated == 8
    return complete, evaluated


def watch(root, pid):
    root = Path(root).resolve()
    state_path = root / 'completion_notification.json'
    state = dict(launcher_pid=pid, watcher_pid=os.getpid(), started_at=timestamp(),
                 status='waiting', mechanism='linux_pidfd', delivery='local_desktop',
                 model_calls=0)

    def save():
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        descriptor = None
    save()
    print(f'Waiting for training launcher PID {pid}; event-driven, no model calls.', flush=True)
    if descriptor is not None:
        try:
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    complete, evaluated = completion(root)
    title = 'MANGO training complete' if complete else 'MANGO training stopped before completion'
    body = (f'All 8 ESM2 models are trained and evaluated. Results: {root}' if complete
            else f'{evaluated}/8 final models evaluated. Check {root / "launcher.log"}.')
    state.update(status='complete' if complete else 'incomplete', finished_at=timestamp(),
                 evaluated_final_models=evaluated, title=title, message=body)
    result = subprocess.run(['notify-send', '--app-name=MANGO', '--expire-time=0',
                             '--urgency=normal' if complete else '--urgency=critical', title, body],
                            capture_output=True, text=True, timeout=20)
    state['notification_submitted'] = result.returncode == 0
    if result.returncode:
        state['notification_error'] = result.stderr.strip()
    save()
    print(json.dumps(state, indent=2), flush=True)
    return 0 if result.returncode == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(watch(args.root, args.pid))
