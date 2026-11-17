"""Completion-driven orchestration retries only infrastructure failures."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from run_affinity_study import Controller


class ControllerTests(unittest.TestCase):
    def controller(self,root):
        controller=Controller.__new__(Controller)
        controller.protocol=dict(implementation_sha256={});controller.root=Path(root)
        controller.config='test.yaml';controller.jobs={};controller.failed=[];controller.decisions={}
        return controller

    def test_transient_retry_and_completed_stage_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.controller(directory);calls=[]
            def start(*args,**kwargs):
                calls.append(args)
                if len(calls)==1:kwargs['stdout'].write('Connection reset by peer')
                return SimpleNamespace(wait=lambda:1 if len(calls)==1 else 0)
            with patch('run_affinity_study.subprocess.Popen',side_effect=start):
                self.assertTrue(controller.stage('train',Path(directory)/'complete.json'))
                self.assertTrue(controller.stage('train',Path(directory)/'complete.json'))
            self.assertEqual(len(calls),2)
            self.assertEqual(controller.jobs['train']['status'],'complete')

    def test_scientific_failure_does_not_retry_or_block_independent_work(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.controller(directory)
            with patch('run_affinity_study.subprocess.Popen',return_value=SimpleNamespace(wait=lambda:1)) as start:
                self.assertFalse(controller.stage('invalid_cohort',Path(directory)/'bad.json'))
                self.assertFalse(controller.stage('invalid_cohort',Path(directory)/'bad.json'))
                self.assertEqual(start.call_count,1)
            with patch('run_affinity_study.subprocess.Popen',return_value=SimpleNamespace(wait=lambda:0)):
                self.assertTrue(controller.stage('independent',Path(directory)/'good.json'))
            self.assertEqual(controller.failed,['invalid_cohort'])

    def test_interruption_terminates_child_group_before_unwinding(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.controller(directory)
            process=SimpleNamespace(pid=12345)
            calls=iter([KeyboardInterrupt(),0])
            def wait(**kwargs):
                result=next(calls)
                if isinstance(result,BaseException):raise result
                return result
            process.wait=wait
            with patch('run_affinity_study.subprocess.Popen',return_value=process),patch('run_affinity_study.os.killpg') as kill:
                with self.assertRaises(KeyboardInterrupt):controller.stage('interrupted',Path(directory)/'result.json')
                self.assertEqual(kill.call_count,1)


if __name__=='__main__':unittest.main()
