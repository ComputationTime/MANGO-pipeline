"""Queue exactly one Codex review after a study process exits, without polling."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import subprocess


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def process_start(pid):
    """Kernel start time prevents waiting on an unrelated, reused PID."""
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    boot=next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    return boot+int(stat[19])/os.sysconf('SC_CLK_TCK')


def wait_for_exit(pid, launched_at):
    try:
        descriptor=os.pidfd_open(pid)
    except ProcessLookupError:
        return 'already_exited'
    try:
        try:started=process_start(pid)
        except FileNotFoundError:return 'already_exited'
        expected=datetime.fromisoformat(launched_at).timestamp()
        if abs(started-expected)>10:
            return 'original_process_exited_pid_reused'
        # No timer, metric reads, subprocess polling, or model calls.
        select.select([descriptor],[],[])
        return 'process_exit_event'
    finally:os.close(descriptor)


def codex_binary():
    home=Path(os.environ.get('CODEX_HOME',Path.home()/'.codex'))
    candidates=[home/'packages/app-server-daemon/current/bin/codex']
    found=shutil.which('codex')
    if found:candidates.append(Path(found))
    for path in candidates:
        if not path.is_file():continue
        result=subprocess.run([str(path),'queue','--help'],capture_output=True,text=True,timeout=15)
        if result.returncode==0 and '--resume' in result.stdout:return str(path)
    raise RuntimeError('A Codex CLI supporting queue --resume is required')


def review_prompt(root):
    return (
        'Completion-triggered review authorized by the user. The unattended MANGO '
        f'study launcher for {root.resolve()} has exited. Review its state.json, '
        'results_report.json, findings.md, fatal.json if present, and completed-stage logs. '
        'Check scientific validity, cohort exclusions, failed/infeasible arms, matched IgLM '
        'comparisons, uncertainty and advancement decisions. Explain the actual outcome '
        'and any root causes to the user in this conversation. Write a review to '
        f'{root.resolve()}/completion_review/assistant_review.md if workspace permissions allow. '
        'This is a review request: do not change the frozen protocol, restart training, '
        'launch new experiments, or send messages to other people. Read experiment logs '
        'only now that the launcher has exited. If reports are missing, diagnose the exit '
        'rather than claiming successful completion.'
    )


def queue_review(binary,thread_id,prompt):
    result=subprocess.run([binary,'queue','--thread',thread_id,'--message',prompt,'--resume'],
                          capture_output=True,text=True,timeout=60)
    output=result.stdout+'\n'+result.stderr
    # queue can exit zero even when resuming fails. Preserve that distinction;
    # never retry automatically and risk enqueueing the same review twice.
    resumed=result.returncode==0 and not any(word in output.lower() for word in ('warning','failed','error:'))
    return dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                resume_confirmed=resumed)


def watch(root,thread_id,ready_fd=None):
    root=Path(root).resolve();directory=root/'completion_review';directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'watcher.json'
    with (directory/'watcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.exists():
            previous=json.loads(state_path.read_text()).get('status')
            if previous in ('review_queued','resume_warning','submission_started','submission_uncertain'):
                return 0 if previous=='review_queued' else 1
        launch=json.loads((root/'launch.json').read_text());binary=codex_binary()
        state=dict(status='waiting',watcher_pid=os.getpid(),study_pid=launch['pid'],thread_id=thread_id,
                   started_at=now(),mechanism='linux_pidfd',model_calls_while_waiting=0,codex=binary)
        save(state_path,state)
        if ready_fd is not None:
            os.write(ready_fd,b'ready\n');os.close(ready_fd)
        try:
            event=wait_for_exit(launch['pid'],launch['started_at'])
            state.update(status='launcher_exited',event=event,launcher_exited_at=now());save(state_path,state)
            prompt=review_prompt(root);(directory/'review_prompt.txt').write_text(prompt+'\n')
            state.update(status='submission_started',submission_started_at=now());save(state_path,state)
            result=queue_review(binary,thread_id,prompt)
            state.update(status='review_queued' if result['resume_confirmed'] else 'resume_warning',
                         submitted_at=now(),submission=result)
            save(state_path,state)
            return 0 if result['resume_confirmed'] else 1
        except Exception as error:
            state.update(status='submission_uncertain' if state['status']=='submission_started' else 'failed',
                         error=repr(error),failed_at=now());save(state_path,state)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--thread-id',required=True)
    p.add_argument('--ready-fd',type=int,help=argparse.SUPPRESS)
    args=p.parse_args();raise SystemExit(watch(args.root,args.thread_id,args.ready_fd))
