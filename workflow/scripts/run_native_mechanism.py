"""Completion-driven, serialized native mechanism pilot with honest data gates."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml
from affinity_study_common import digest,freeze,read_json,write_json
from native_mechanism_common import arms,CONTRASTS


class Controller:
    def __init__(self,config,smoke=False):
        self.config=os.path.relpath(Path(config).resolve(),Path.cwd());self.cfg=yaml.safe_load(Path(config).read_text())
        if smoke:
            self.cfg.update(experiment_root='artifacts/experiments/native_mechanism_smoke_v2',smoke=True,seeds=[100],
                pretraining_updates=4,adaptation_updates=4,diagnostic_pairs_per_target=8,bootstrap_repeats=100)
            root=Path(self.cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
            self.config=str((root/'settings.yaml').resolve());value=yaml.safe_dump(self.cfg,sort_keys=False)
            if Path(self.config).exists() and Path(self.config).read_text()!=value:raise ValueError('Smoke settings changed')
            Path(self.config).write_text(value)
        self.root=Path(self.cfg['experiment_root']);self.root.mkdir(parents=True,exist_ok=True)
        self.jobs={};self.failed=[]
        files=[Path('workflow/scripts')/n for n in ('native_mechanism_worker.py','native_mechanism_common.py',
            'native_mechanism_embeddings.py','prepare_native_schedules.py','prepare_native_smoke.py','prepare_native_pilot.py',
            'run_native_mechanism.py','report_native_mechanism.py','controlled_antigen_model.py','short_affinity_model.py',
            'short_affinity_common.py','antigen_diagnostics.py','iglm_baseline.py','pairing_prefix_score_cache.py',
            'pairing_score_cache.py','affinity_study_common.py','device_common.py','chain_examples.py','embed_common.py',
            'embed_antigen_esm2.py','short_affinity_embeddings.py')]
        files += [Path('workflow/native_mechanism.smk'),Path(self.config)]
        files += [Path('workflow/envs')/n for n in ('model.yaml','embed_esm2.yaml','openalphaseq_data.yaml')]
        self.protocol=dict(settings=self.cfg,arms=arms(),contrasts=CONTRASTS,
            implementation_sha256={str(p):digest(p) for p in files},
            checkpoint_selection='fixed final update; no validation selection or automatic extension',
            status='exploratory pilot, not confirmation',
            uncertainty='component bootstrap conditional on realized seeds; separate seed SD; no p-values')
        freeze(self.root/'training_protocol.json',self.protocol)

    def state(self,status='running'):
        write_json(self.root/'state.json',dict(status=status,pid=os.getpid(),jobs=self.jobs,failed_stages=self.failed,updated_at=time.time()))

    def stage(self,name,target,timeout=None):
        for path,value in self.protocol['implementation_sha256'].items():
            if digest(path)!=value:raise ValueError('Frozen implementation changed: '+path)
        directory=self.root/'controller_logs';directory.mkdir(exist_ok=True);log=directory/f'{name}.log'
        env=dict(os.environ,MANGO_NATIVE_MECHANISM_CONFIG=self.config)
        command=['.snakemake/gpu-driver/bin/snakemake','-s','workflow/native_mechanism.smk','--profile','workflow/profiles/gpu',
            '--rerun-incomplete','--rerun-triggers','mtime','--',str(target)]
        started=time.monotonic()
        with log.open('w') as handle:
            child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=child.wait(timeout=timeout)
            except BaseException as error:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                if not isinstance(error,subprocess.TimeoutExpired):raise
                code=124
        success=code==0 and Path(target).exists()
        self.jobs[name]=dict(status='complete' if success else 'incomplete' if code==124 else 'failed',
            returncode=code,log=str(log),wall_seconds=time.monotonic()-started)
        if not success:self.failed.append(name)
        self.state();return success

    def run(self):
        self.state()
        if not self.stage('audit',self.root/'cohort_audit.json',self.cfg['audit_timeout_seconds']):return self.finish()
        if not read_json(self.root/'cohort_audit.json')['feasible']:return self.finish(infeasible=True)
        if not self.stage('schedules',self.root/'schedules_complete.json'):return self.finish()
        features=self.stage('embeddings',self.root/'embeddings_complete.json',900)
        for seed in self.cfg['seeds']:
            for arm,spec in arms().items():
                if spec['branch'] and not features:continue
                if spec['source'] and not (self.root/'runs'/f'{spec["source"]}_s{seed}'/'complete.json').exists():continue
                ident=f'{arm}_s{seed}'
                if self.stage(ident,self.root/'runs'/ident/'complete.json',self.cfg['fit_timeout_seconds']):
                    self.stage('external_'+ident,self.root/'runs'/ident/'affinity_complete.json',self.cfg['fit_timeout_seconds'])
        return self.finish()

    def finish(self,infeasible=False):
        self.state('reporting');self.stage('report',self.root/'results_report.json')
        status='data_infeasible' if infeasible else 'incomplete' if self.failed else 'complete'
        self.state(status)
        print(str(self.root/'findings.md'),flush=True)
        return 2 if infeasible else int(bool(self.failed))


def main(config,smoke=False):
    controller=Controller(config,smoke)
    def stop(signum,frame):raise InterruptedError('Experiment stopped')
    signal.signal(signal.SIGTERM,stop)
    with (controller.root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return controller.run()
        except BaseException as error:
            write_json(controller.root/'fatal.json',dict(error=repr(error)))
            controller.state('interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/native_mechanism_pilot_v2.yaml');p.add_argument('--smoke',action='store_true')
    raise SystemExit(main(**vars(p.parse_args())))
