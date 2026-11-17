"""Completion-driven short tests with immutable code and explicit resource gates."""
import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml

from affinity_study_common import digest,freeze,read_json,write_json
from short_affinity_common import ARMS


def terminate_tree(pid):
    import psutil
    try:parent=psutil.Process(pid);children=parent.children(recursive=True)
    except psutil.NoSuchProcess:return
    for process in children[::-1]+[parent]:
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(children+[parent],timeout=5)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass


class Controller:
    def __init__(self,config):
        self.config=str(Path(config).resolve());self.cfg=yaml.safe_load(Path(config).read_text());self.root=Path(self.cfg['experiment_root'])
        self.root.mkdir(parents=True,exist_ok=True);self.jobs={};self.tests={};self.used_gpu=0.;self.failed=[]
        old=read_json(Path(self.cfg['reference_root'])/'protocol.json')['implementation_sha256']
        for path,value in old.items():
            if digest(path)!=value:raise ValueError('Frozen v1 code changed: '+path)
        files=[Path('workflow/scripts')/name for name in ('short_affinity_common.py','prepare_short_affinity.py','short_affinity_model.py',
            'short_affinity_embeddings.py','short_affinity_worker.py','run_short_affinity.py','report_short_affinity.py',
            'controlled_antigen_common.py','controlled_antigen_model.py','prepare_controlled_antigen.py',
            'pairing_prefix_score_cache.py','pairing_score_cache.py','prepare_abrank_transfer.py','asd_scaling_studies.py','embed_antigen_esm2.py','embed_common.py')]
        files.extend(Path(path) for path in old)
        files.extend([Path('workflow/short_affinity.smk'),Path('docs/short_affinity_tests_v3.md'),Path('run_gpu.sh')])
        files.extend(Path('workflow/envs')/name for name in ('model.yaml','embed_esm2.yaml','analysis_anarci.yaml','openalphaseq_data.yaml'))
        self.protocol=dict(version=3,settings=self.cfg,arms=ARMS,implementation_sha256={str(p):digest(p) for p in files},
            stage_policy='Completion waits; no intermediate metric review. Screening only; no automatic confirmation.',
            parameter_control='IgLM FFN width matched within 0.1%; shared native tensors copied and extra output weights zero-initialized',
            biological_gate='20 components/100 heavy examples per panel; original holdouts never trained')
        freeze(self.root/'protocol.json',self.protocol)

    def state(self,status='running'):
        write_json(self.root/'state.json',dict(status=status,pid=os.getpid(),jobs=self.jobs,tests=self.tests,
            gpu_budget_used_seconds=self.used_gpu,failed_stages=self.failed,updated_at=time.time()))

    def stage(self,name,target,limit,gpu=False,startup=45):
        for path,value in self.protocol['implementation_sha256'].items():
            if digest(path)!=value:raise ValueError('Frozen implementation changed: '+path)
        if gpu:limit=min(limit,self.cfg['budgets']['total_gpu']-self.used_gpu)
        if limit<=0:
            self.jobs[name]=dict(status='incomplete',reason='global_gpu_budget');self.state();return False
        logs=self.root/'controller_logs';logs.mkdir(exist_ok=True);log=logs/f'{name}.log'
        command=['.snakemake/gpu-driver/bin/snakemake','-s','workflow/short_affinity.smk','--profile','workflow/profiles/gpu',
            '--rerun-incomplete','--rerun-triggers','mtime','--configfile',self.config,'--',str(target)]
        started=time.monotonic()
        with log.open('w') as handle:
            child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,MANGO_SHORT_CONFIG=self.config),start_new_session=True)
            try:
                timeout=min(limit+startup,self.cfg['budgets']['total_gpu']-self.used_gpu) if gpu else limit+startup
                code=child.wait(timeout=timeout)
                value=dict(status='complete' if code==0 else 'failed',returncode=code,log=str(log))
            except subprocess.TimeoutExpired:
                terminate_tree(child.pid);child.wait();value=dict(status='incomplete',reason='stage_time_limit',log=str(log))
            except BaseException:terminate_tree(child.pid);child.wait();raise
        elapsed=time.monotonic()-started
        if gpu:self.used_gpu+=elapsed
        value['wall_seconds']=elapsed
        if value['status']=='failed':self.failed.append(name)
        self.jobs[name]=value;self.state();return value['status']=='complete'

    def run(self):
        self.state();selection=self.stage('selection',self.root/'diagnostic_selection.json',120)
        remaining=self.cfg['budgets']['diagnostics']
        if selection:
            for seed in self.cfg['seeds']:
                for kind in ('base','iglm'):
                    name=f'pilot_{kind}_s{seed}';before=self.used_gpu
                    self.stage('diagnostic_'+name,self.root/'diagnostics'/f'{name}.complete.json',remaining,gpu=True,startup=0)
                    remaining-=self.used_gpu-before
        diagnostics=list((self.root/'diagnostics').glob('*.complete.json'))
        self.tests['2']=dict(status='complete' if len(diagnostics)==2*len(self.cfg['seeds']) else 'incomplete',completed_checkpoints=len(diagnostics))
        okay=self.stage('audit',self.root/'cohort_audit.json',self.cfg['budgets']['audit'],startup=0)
        audit=read_json(self.root/'cohort_audit.json') if okay else None
        self.tests['1']=dict(status='complete' if okay else 'incomplete',training_feasible=bool(audit and audit['feasible']))
        if not audit or not audit['feasible']:
            for number in (3,4,5,6):self.tests[str(number)]=dict(status='infeasible' if audit else 'blocked',reason=audit.get('reason') if audit else 'audit_incomplete')
            return self.finish()
        okay=self.stage('embeddings',self.root/'embeddings_complete.json',self.cfg['budgets']['embeddings'],gpu=True,startup=0)
        if okay:okay=self.stage('regions',self.root/'regions.json',300)
        if not okay:
            for number in (3,4,5,6):self.tests[str(number)]=dict(status='blocked',reason='feature_or_region_preparation_incomplete')
            return self.finish()
        ranking=read_json(self.root/'ranking_audit.json')
        for arm,values in ARMS.items():
            number=str(values['test'])
            if values['test']==6 and not ranking['feasible']:
                self.tests[number]=dict(status='infeasible',reason=ranking['reason']);continue
            for seed in self.cfg['seeds']:
                ident=f'{arm}_s{seed}';spec=dict(values,id=ident,seed=seed,arm=arm)
                freeze(self.root/'specs'/f'{ident}.json',spec)
                complete=self.stage(ident,self.root/'runs'/ident/'status.json',self.cfg['budgets']['fit'],gpu=True)
                if complete:
                    stage=read_json(self.root/'runs'/ident/'status.json')
                    if stage['status']!='complete':self.jobs[ident].update(status='incomplete',reason=stage['reason'])
            relevant=[self.jobs.get(f'{a}_s{s}',{}) for a,v in ARMS.items() if v['test']==values['test'] for s in self.cfg['seeds']]
            self.tests[number]=dict(status='complete' if all(r.get('status')=='complete' for r in relevant) else 'incomplete')
            self.state()
        return self.finish()

    def finish(self):
        self.state('reporting');self.stage('report',self.root/'results_report.json',300)
        incomplete=self.failed or any(v['status'] in ('incomplete','blocked','failed') for v in self.tests.values())
        self.state('incomplete' if incomplete else 'complete');return int(bool(incomplete))


def main(config,smoke=False):
    cfg=yaml.safe_load(Path(config).read_text())
    if smoke:
        cfg.update(experiment_root='artifacts/experiments/affinity_short_smoke_v3',smoke=True,seeds=[0],updates=16,
            diagnostic_pairs_per_target=2,bootstrap_repeats=100,minimum_groups=1,minimum_heavy_examples=1)
        root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True);cfg['verified_affinity_records']=str(root/'synthetic_affinities.csv')
        config=str(root/'settings.yaml');content=yaml.safe_dump(cfg,sort_keys=False)
        if Path(config).exists() and Path(config).read_text()!=content:raise ValueError('Smoke configuration changed')
        if not Path(config).exists():Path(config).write_text(content)
    root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    def stop(sig,frame):raise InterruptedError('Short-test controller terminated')
    signal.signal(signal.SIGTERM,stop)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=Controller(config)
        try:return controller.run()
        except Exception as error:
            write_json(root/'fatal.json',dict(error=repr(error)));controller.failed.append('controller');controller.finish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/short_affinity_tests.yaml');p.add_argument('--smoke',action='store_true')
    raise SystemExit(main(**vars(p.parse_args())))
