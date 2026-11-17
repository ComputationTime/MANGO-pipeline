"""Completion-driven primary and secondary ablations; never tunes on external affinity."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from pathlib import Path
import fcntl
import json
import os
import signal
import shutil
import subprocess
import time

from affinity_study_common import digest,freeze,read_csv,read_json,write_json,retry_kind
from controlled_antigen_common import PRIMARY,SECONDARY,score_arrays,crossed_interval
from affinity_study_common import holm


class Controller:
    def __init__(self,config):
        import yaml
        self.config=str(Path(config).resolve());self.settings=yaml.safe_load(Path(config).read_text())
        self.root=Path(self.settings['experiment_root']);self.root.mkdir(parents=True,exist_ok=True)
        self.jobs={};self.failed=[];self.decisions={}
        files=[(Path('experiments/scripts') if (Path('experiments/scripts')/name).exists() else Path('workflow/scripts'))/name for name in ('controlled_antigen_common.py','controlled_antigen_model.py',
            'controlled_antigen_worker.py','controlled_antigen_embeddings.py','prepare_controlled_antigen.py','run_controlled_antigen.py','report_controlled_antigen.py',
            'affinity_study_worker.py','affinity_study_common.py','prepare_affinity_study.py','antigen_diagnostics.py',
            'iglm_baseline.py','train_antigen_diagnostic.py','pairing_prefix_score_cache.py','pairing_score_cache.py',
            'chain_examples.py','asd_transfer_embeddings.py','number_diagnostic_regions.py','prepare_asd_transfer.py',
            'prepare_abrank_transfer.py','model_common.py','asd_scaling_studies.py','fixed_position_prefix.py','device_common.py')]
        files.extend([Path('experiments/workflow/controlled_antigen.smk'),Path('mango/utils/mango_utils.py')])
        files.extend(Path('workflow/envs')/name for name in ('model.yaml','embed_esm2.yaml','analysis_anarci.yaml','openalphaseq_data.yaml'))
        self.protocol=dict(version=2,settings=self.settings,primary_contrasts=PRIMARY,secondary_contrasts=SECONDARY,
            implementation_sha256={str(p):digest(p) for p in files},
            endpoint='token-weighted heavy NLL; CDR/CDR3 secondary; external affinities exploratory',
            uncertainty='paired crossed bootstrap of matched seed and biological-component axes',
            checkpoint_selection='minimum ordinary validation H/L token-weighted NLL; two epochs per stage')
        freeze(self.root/'protocol.json',self.protocol)

    def state(self,status='running'):
        write_json(self.root/'state.json',dict(status=status,pid=os.getpid(),jobs=self.jobs,failed_stages=self.failed,
            decisions=self.decisions,updated_at=time.time()))

    def stage(self,name,target):
        if name in self.jobs:return self.jobs[name]['status']=='complete'
        for path,value in self.protocol['implementation_sha256'].items():
            if digest(path)!=value:raise ValueError('Frozen code changed: '+path)
        if shutil.disk_usage(self.root).free<10*2**30:raise RuntimeError('Less than 10 GiB free disk')
        logs=self.root/'controller_logs';logs.mkdir(exist_ok=True)
        env=dict(os.environ,MANGO_CONTROLLED_CONFIG=self.config)
        cmd=['.snakemake/gpu-driver/bin/snakemake','-s','experiments/workflow/controlled_antigen.smk','--profile','workflow/profiles/gpu',
             '--rerun-incomplete','--rerun-triggers','mtime','--configfile',self.config,'--',str(target)]
        for attempt in range(2):
            log=logs/f'{name}_attempt{attempt}.log'
            with log.open('w') as handle:
                child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:code=child.wait()
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            if code==0:
                self.jobs[name]=dict(status='complete',log=str(log));self.state();return True
            reason=retry_kind(log.read_text())
            if attempt==0 and reason=='transient':continue
            self.jobs[name]=dict(status='failed',log=str(log),reason=reason or 'stage_error');self.failed.append(name);self.state();return False

    def spec(self,phase,study,arm,seed):
        ident=f'{phase}_{study}_{arm}_s{seed}'
        branch=arm!='iglm';frozen=arm.startswith('frozen_')
        memory=arm.split('_')[-1] if branch else 'none'
        source=f'{phase}_primary_iglm_s{seed}' if study=='secondary' else None
        spec=dict(id=ident,phase=phase,study=study,arm=arm,seed=seed,branch=branch,frozen=frozen,memory=memory,source=source)
        freeze(self.root/'specs'/f'{ident}.json',spec)
        return ident if self.stage(ident,self.root/'runs'/ident/'complete.json') else None

    def family(self,phase,study,seeds):
        arms=['iglm','correct','constant','shuffled'] if study=='primary' else [f'{update}_{context}'
            for update in ('frozen','joint') for context in ('correct','shuffled','constant')]+['iglm']
        for seed in seeds:
            if study=='secondary' and not self.spec(phase,'primary','iglm',seed):continue
            for arm in arms:self.spec(phase,study,arm,seed)

    def comparisons(self,phase,study,seeds):
        contrasts=PRIMARY if study=='primary' else SECONDARY;tables={}
        for seed in seeds:
            tables[seed]={}
            for path in (self.root/'runs').glob(f'{phase}_{study}_*_s{seed}/complete.json'):
                done=read_json(path)
                if digest(path.parent/'scores.csv')!=done['scores_sha256']:raise ValueError('Score provenance changed')
                tables[seed][done['spec']['arm']]=read_csv(path.parent/'scores.csv')
        results={}
        for name,contrast in contrasts.items():
            try:
                losses,tokens=score_arrays(tables,contrast)
                result=crossed_interval(losses,tokens,self.settings['bootstrap_repeats'],self.settings['seed'])
            except ValueError as error:
                result=dict(estimate=None,pvalue=1.,error=str(error),low=None,high=None)
            results[name]=result
        return results

    def run(self):
        self.state()
        # Inference repair is independent of whether new training is feasible.
        selected=read_json(Path(self.settings['reference_root'])/'selection.json')['external_runs']
        if self.settings.get('smoke'):selected=selected[:1]
        for run in selected:self.stage('amendment_'+run,self.root/'amendment_v1'/f'{run}.complete.json')
        if not self.stage('audit',self.root/'cohort_audit.json'):return self.finish()
        audit=read_json(self.root/'cohort_audit.json')
        if not audit['feasible']:
            self.decisions['data']=dict(status='infeasible',audit=audit);return self.finish()
        if not self.stage('embeddings',self.root/'embeddings_complete.json'):return self.finish()
        if not self.stage('regions',self.root/'regions.json'):return self.finish()
        seeds=self.settings['pilot_seeds']
        self.family('pilot','primary',seeds);self.family('pilot','secondary',seeds)
        selected_runs=[]
        for study in ('primary','secondary'):
            pilot=self.comparisons('pilot',study,seeds)
            keys=['pairing'] if study=='primary' else ['pairing_frozen','pairing_joint']
            advance=any(pilot[k].get('estimate') is not None and pilot[k]['estimate']>=self.settings['minimum_improvement']
                and sum(v>0 for v in pilot[k]['seed_points'])>=2 for k in keys)
            self.decisions[study]=dict(pilot=pilot,advance=advance)
            phase='pilot'
            if advance and not self.settings.get('smoke'):
                confirm_seeds=self.settings['confirmation_seeds'];self.family('confirmation',study,confirm_seeds)
                results=self.comparisons('confirmation',study,confirm_seeds)
                reject=holm({k:v['pvalue'] for k,v in results.items()},self.settings[study+'_alpha'])
                self.decisions[study]['confirmation']=dict(results=results,holm=reject,
                    supported={k:bool(reject[k] and v.get('low') is not None and v['low']>0
                        and v['estimate']>=self.settings['minimum_improvement']) for k,v in results.items()})
                phase='confirmation'
            selected_runs.extend(p.parent.name for p in (self.root/'runs').glob(f'{phase}_{study}_*/complete.json'))
            self.state()
        selection=dict(runs=sorted(selected_runs),decisions=self.decisions,external_labels_used_for_selection=False)
        freeze(self.root/'selection.json',selection)
        for run in selected_runs:self.stage('external_'+run,self.root/'external'/f'{run}.complete.json')
        return self.finish()

    def finish(self):
        self.state('reporting');self.stage('report',self.root/'results_report.json')
        self.state('incomplete' if self.failed else 'complete');return int(bool(self.failed))


def main(config,smoke=False):
    import yaml
    cfg=yaml.safe_load(Path(config).read_text())
    if smoke:
        cfg.update(experiment_root='artifacts/experiments/controlled_antigen_smoke_v2',smoke=True,
                   pilot_seeds=[0],confirmation_seeds=[10],epochs=1,bootstrap_repeats=100)
        root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
        config=str(root/'settings.yaml');content=yaml.safe_dump(cfg,sort_keys=False)
        if Path(config).exists() and Path(config).read_text()!=content:raise ValueError('Frozen smoke settings changed')
        Path(config).write_text(content)
    root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):raise InterruptedError('Controller terminated')
    signal.signal(signal.SIGTERM,stop)
    with (root/'controller.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=Controller(config)
        try:return controller.run()
        except Exception as error:
            write_json(root/'fatal.json',dict(error=repr(error)));controller.failed.append('controller');controller.finish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='experiments/config/controlled_antigen_study.yaml');p.add_argument('--smoke',action='store_true')
    raise SystemExit(main(**vars(p.parse_args())))
