"""ASD preparation and independent-arm failure handling in the GPU launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class TransferLauncherTests(unittest.TestCase):
    def launch(self, directory, fail=False, mode='asd-transfer'):
        root = Path(directory)
        (root / 'run_gpu.sh').write_text((ROOT / 'run_gpu.sh').read_text())
        config = root / 'config'
        config.mkdir()
        study = dict(experiment_root='artifacts/custom_asd', cohort_dir='artifacts/custom_asd/cohort',
                     controls=['esm2', 'control_cross_attention_esm2'])
        filename = 'asd_transfer.yaml'
        if mode in {'asd-loss-comparison', 'asd-prototype'}:
            filename = 'asd_loss_comparison.yaml' if mode == 'asd-loss-comparison' else 'asd_prototype.yaml'
            study.update(objectives={'cross_entropy': {}, 'focal': {}},
                         architectures={'mango': '', 'cross_attention': ''})
        elif mode == 'asd-prototype-iglm':
            filename = 'asd_prototype_iglm.yaml'
            study['mango_root'] = 'artifacts/completed_mango'
        elif mode in {'antigen-diagnostics', 'antigen-diagnostic-training'}:
            filename = mode.replace('-', '_') + '.yaml'
            study['arms' if mode.endswith('-training') else 'probes'] = {'first': {}, 'second': {}}
        elif mode in {'asd-scaling', 'asd-scaling-8k', 'asd-scaling-16k', 'asd-scaling-fixed-positions'}:
            filename = mode.replace('-', '_') + '.yaml'
            study['runs'] = {'first': {}, 'second': {}}
        (config / filename).write_text(yaml.safe_dump(study, sort_keys=False))
        binaries = root / 'bin'
        binaries.mkdir()
        for name in ('conda', 'nvidia-smi'):
            path = binaries / name
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o755)
        driver = root / '.snakemake/gpu-driver/bin'
        driver.mkdir(parents=True)
        stub = (
            f'#!{sys.executable}\n'
            'import json,os,sys\n'
            "if sys.argv[1:2]==['-c']: os.execv(sys.executable,[sys.executable]+sys.argv[1:])\n"
            "with open(os.environ['TRANSFER_CALLS'],'a') as h: h.write(json.dumps(sys.argv)+'\\n')\n"
            "fail=os.environ['TRANSFER_FAIL']=='1'\n"
            "targets={'artifacts/custom_asd/asd/runs/esm2/eval.json','artifacts/custom_asd/sabdab2_scratch/cross_entropy/mango/eval.json','artifacts/custom_asd/sabdab2_scratch/eval.json','artifacts/custom_asd/probes/first/complete.json','artifacts/custom_asd/first/native/complete.json','artifacts/custom_asd/runs/first/eval.json'}\n"
            "report=any(a.startswith('workflow/scripts/report_') for a in sys.argv) and '--require-complete' in sys.argv\n"
            "report=report or (any(a in {'workflow/antigen_diagnostics.smk','workflow/antigen_diagnostic_training.smk','workflow/asd_scaling.smk'} for a in sys.argv) and '--keep-going' in sys.argv)\n"
            "sys.exit(1 if fail and (targets.intersection(sys.argv) or report) else 0)\n"
        )
        for name in ('snakemake', 'python'):
            path = driver / name
            path.write_text(stub)
            path.chmod(0o755)
        calls = root / 'calls.jsonl'
        environment = dict(os.environ, PATH=f'{binaries}:{os.environ["PATH"]}',
                           CONDA_EXE=str(binaries / 'conda'), TRANSFER_CALLS=str(calls),
                           TRANSFER_FAIL='1' if fail else '0')
        result = subprocess.run(['bash', str(root / 'run_gpu.sh'), mode],
                                env=environment, capture_output=True, text=True, timeout=30)
        return result, [json.loads(line) for line in calls.read_text().splitlines()]

    def test_asd_prepares_configured_cohort_then_runs_both_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0][1:], ['workflow/scripts/prepare_asd_transfer.py', '--output', 'artifacts/custom_asd/cohort'])
        jobs = [call for call in calls if call[0].endswith('/snakemake')]
        self.assertEqual(len(jobs), 5)
        self.assertTrue(all('workflow/asd_transfer.smk' in call for call in jobs))
        self.assertEqual([call[-1] for call in jobs[:4]], [
            'artifacts/custom_asd/asd/runs/esm2/eval.json',
            'artifacts/custom_asd/asd/runs/control_cross_attention_esm2/eval.json',
            'artifacts/custom_asd/sabdab2/runs/esm2/eval.json',
            'artifacts/custom_asd/sabdab2/runs/control_cross_attention_esm2/eval.json'])
        self.assertIn('--require-complete', calls[-1])

    def test_failed_arm_does_not_block_later_controls_and_report_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp, fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('continuing with other controls', result.stderr)
        self.assertEqual(sum(call[0].endswith('/snakemake') for call in calls), 5)
        self.assertIn('workflow/scripts/report_abrank_transfer.py', calls[-1])

    def test_loss_comparison_runs_scratch_first_then_all_matching_transfer_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp, mode='asd-loss-comparison')
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = [call for call in calls if call[0].endswith('/snakemake')]
        self.assertEqual(len(jobs), 13)
        endpoints = [call[-1] for call in jobs[:-1]]
        self.assertTrue(all('/sabdab2_scratch/' in p for p in endpoints[:4]))
        self.assertEqual(len(set(endpoints)), 12)
        for loss in ('cross_entropy', 'focal'):
            for arch in ('mango', 'cross_attention'):
                for phase in ('asd', 'sabdab2_scratch', 'sabdab2_finetuned'):
                    self.assertIn(f'artifacts/custom_asd/{phase}/{loss}/{arch}/eval.json', endpoints)
        self.assertTrue(all('workflow/asd_loss_comparison.smk' in call for call in jobs))
        self.assertIn('--require-complete', calls[-1])

    def test_loss_comparison_failure_keeps_queue_running_and_reports_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp, fail=True, mode='asd-loss-comparison')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(call[0].endswith('/snakemake') for call in calls), 13)
        self.assertIn('continuing with other models', result.stderr)
        self.assertIn('workflow/scripts/report_asd_loss_comparison.py', calls[-1])

    def test_prototype_prepares_small_cohort_and_uses_its_config_for_every_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp, mode='asd-prototype')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0][1:], ['workflow/scripts/prepare_asd_prototype.py', '--config', 'config/asd_prototype.yaml'])
        jobs = [call for call in calls if call[0].endswith('/snakemake')]
        self.assertEqual(len(jobs), 13)
        self.assertTrue(all('config/asd_prototype.yaml' in call for call in jobs))

    def test_iglm_extension_runs_both_routes_without_rerunning_mango(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp, mode='asd-prototype-iglm')
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = [call for call in calls if call[0].endswith('/snakemake')]
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all('workflow/asd_prototype_iglm.smk' in call for call in jobs))
        self.assertEqual([call[-1] for call in jobs[:3]], [
            f'artifacts/custom_asd/{phase}/eval.json'
            for phase in ('sabdab2_scratch', 'sabdab2_finetuned', 'asd')])
        self.assertTrue(all('prepare_' not in arg for call in calls for arg in call))
        self.assertIn('--require-complete', calls[-1])

    def test_iglm_extension_failure_preserves_independent_route_and_partial_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.launch(tmp, fail=True, mode='asd-prototype-iglm')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(call[0].endswith('/snakemake') for call in calls), 4)
        self.assertIn('continuing with other stages', result.stderr)
        self.assertIn('workflow/scripts/report_prototype_iglm.py', calls[-1])

    def test_diagnostic_launchers_run_independent_targets_and_propagate_failure(self):
        for mode in ('antigen-diagnostics', 'antigen-diagnostic-training'):
            for fail in (False, True):
                with self.subTest(mode=mode, fail=fail), tempfile.TemporaryDirectory() as tmp:
                    result, calls = self.launch(tmp, fail=fail, mode=mode)
                    jobs = [call for call in calls if call[0].endswith('/snakemake')]
                    self.assertEqual(len(jobs), 3)
                    self.assertTrue(all('workflow/' + mode.replace('-', '_') + '.smk' in call for call in jobs))
                    self.assertEqual(result.returncode != 0, fail, result.stderr)
                    if fail:
                        self.assertIn('continuing with other', result.stderr)

    def test_scaling_prepares_once_and_preserves_independent_runs_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                result, calls = self.launch(tmp, fail=fail, mode='asd-scaling')
                self.assertEqual(result.returncode != 0, fail, result.stderr)
                self.assertEqual(calls[0][1:], ['workflow/scripts/prepare_asd_scaling.py'])
                jobs = [call for call in calls if call[0].endswith('/snakemake')]
                self.assertEqual(len(jobs), 3)
                self.assertEqual([call[-1] for call in jobs[:2]],
                                 ['artifacts/custom_asd/runs/first/eval.json', 'artifacts/custom_asd/runs/second/eval.json'])
                self.assertTrue(all('workflow/asd_scaling.smk' in call for call in jobs))
                statuses = [call for call in calls if 'workflow/scripts/report_asd_scaling.py' in call]
                self.assertEqual(len(statuses), 3 if fail else 2)

    def test_8k_scaling_uses_its_own_config_and_continues_after_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                result, calls = self.launch(tmp, fail=fail, mode='asd-scaling-8k')
                self.assertEqual(result.returncode != 0, fail, result.stderr)
                self.assertEqual(calls[0][1:], ['workflow/scripts/prepare_asd_scaling_8k.py'])
                jobs = [call for call in calls if call[0].endswith('/snakemake')]
                self.assertEqual(len(jobs), 3)
                self.assertEqual([call[-1] for call in jobs[:2]],
                                 ['artifacts/custom_asd/runs/first/eval.json', 'artifacts/custom_asd/runs/second/eval.json'])
                statuses = [call for call in calls if 'workflow/scripts/report_asd_scaling_8k.py' in call]
                self.assertEqual(len(statuses), 3 if fail else 2)
                self.assertTrue(all(call[call.index('--config') + 1] == 'config/asd_scaling_8k.yaml'
                                    for call in statuses))

    def test_16k_scaling_keeps_new_runs_independent_and_uses_frozen_reference_report(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                result, calls = self.launch(tmp, fail=fail, mode='asd-scaling-16k')
                self.assertEqual(result.returncode != 0, fail, result.stderr)
                self.assertEqual(calls[0][1:], ['workflow/scripts/prepare_asd_scaling_16k.py'])
                jobs = [call for call in calls if call[0].endswith('/snakemake')]
                self.assertEqual(len(jobs), 3)
                self.assertEqual([call[-1] for call in jobs[:2]],
                                 ['artifacts/custom_asd/runs/first/eval.json', 'artifacts/custom_asd/runs/second/eval.json'])
                statuses = [call for call in calls if 'workflow/scripts/report_asd_scaling_8k.py' in call]
                self.assertEqual(len(statuses), 3 if fail else 2)
                self.assertTrue(all(call[call.index('--config') + 1] == 'config/asd_scaling_16k.yaml'
                                    for call in statuses))

    def test_fixed_position_extension_prepares_its_config_and_keeps_runs_independent(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                result, calls = self.launch(tmp, fail=fail, mode='asd-scaling-fixed-positions')
                self.assertEqual(result.returncode != 0, fail, result.stderr)
                self.assertEqual(calls[0][1:], ['workflow/scripts/prepare_asd_scaling_16k.py',
                    '--config', 'config/asd_scaling_fixed_positions.yaml'])
                jobs = [call for call in calls if call[0].endswith('/snakemake')]
                self.assertEqual(len(jobs), 3)
                self.assertEqual([call[-1] for call in jobs[:2]],
                                 ['artifacts/custom_asd/runs/first/eval.json', 'artifacts/custom_asd/runs/second/eval.json'])
                reports = [call for call in calls if 'workflow/scripts/report_asd_scaling_8k.py' in call]
                self.assertTrue(all(call[call.index('--config') + 1] == 'config/asd_scaling_fixed_positions.yaml'
                                    for call in reports))


if __name__ == '__main__':
    unittest.main()
