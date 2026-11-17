"""Exit-triggered review: PID reuse, one-shot delivery and resume warning handling."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow/scripts'))
from review_on_study_exit import process_start,queue_review,review_prompt,wait_for_exit,watch


class ExitReviewTests(unittest.TestCase):
    def test_pidfd_wait_returns_only_after_child_exits(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
        stamp=datetime.fromtimestamp(process_start(child.pid),timezone.utc).isoformat()
        self.assertEqual(wait_for_exit(child.pid,stamp),'process_exit_event')
        self.assertEqual(child.wait(),0)

    def test_reused_pid_is_not_waited_on(self):
        with patch('review_on_study_exit.os.pidfd_open',return_value=123),patch('review_on_study_exit.os.close'),\
             patch('review_on_study_exit.process_start',return_value=100),patch('review_on_study_exit.select.select') as wait:
            self.assertEqual(wait_for_exit(12,'2026-11-05T00:00:00+00:00'),'original_process_exited_pid_reused')
            wait.assert_not_called()

    def test_zero_exit_with_resume_warning_is_not_reported_as_success(self):
        response=subprocess.CompletedProcess([],0,stdout='Queued',stderr='Warning: failed to resume session')
        with patch('review_on_study_exit.subprocess.run',return_value=response) as run:
            result=queue_review('/bin/codex','thread-id','review completed study')
            self.assertFalse(result['resume_confirmed']);self.assertEqual(run.call_count,1)
            self.assertIn('--resume',run.call_args.args[0])

    def test_review_prompt_does_not_authorize_new_training(self):
        prompt=review_prompt(Path('/tmp/study'))
        self.assertIn('has exited',prompt)
        self.assertIn('do not change the frozen protocol',prompt)
        self.assertIn('restart training',prompt)

    def test_restarting_watcher_does_not_enqueue_a_second_review(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'launch.json').write_text(json.dumps(dict(pid=123,started_at='2026-11-05T00:00:00+00:00')))
            result=dict(returncode=0,stdout='Queued',stderr='',resume_confirmed=True)
            with patch('review_on_study_exit.codex_binary',return_value='/bin/codex'),\
                 patch('review_on_study_exit.wait_for_exit',return_value='process_exit_event'),\
                 patch('review_on_study_exit.queue_review',return_value=result) as queue:
                self.assertEqual(watch(root,'thread-id'),0)
                self.assertEqual(watch(root,'thread-id'),0)
                self.assertEqual(queue.call_count,1)
            state=json.loads((root/'completion_review/watcher.json').read_text())
            self.assertEqual(state['status'],'review_queued')

    def test_uncertain_submission_is_not_repeated(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'completion_review';path.mkdir()
            (path/'watcher.json').write_text(json.dumps(dict(status='submission_started')))
            with patch('review_on_study_exit.queue_review') as queue:
                self.assertEqual(watch(Path(directory),'thread-id'),1)
                queue.assert_not_called()


if __name__=='__main__':unittest.main()
