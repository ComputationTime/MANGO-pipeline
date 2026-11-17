"""Completion-driven controller: no polling, metric tailing, or adaptive searches."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from affinity_study_common import (INTERVENTIONS, advancement, digest, freeze, holm,
    paired_group_bootstrap, read_csv, read_json, retry_kind, write_json)


class Controller:
    def __init__(self,config):
        import yaml
        self.config=str(Path(config).resolve());self.settings=yaml.safe_load(Path(config).read_text())
        self.root=Path(self.settings['experiment_root']);self.root.mkdir(parents=True,exist_ok=True)
        self.jobs={};self.decisions={};self.failed=[]
        files=[(Path('experiments/scripts') if (Path('experiments/scripts')/name).exists() else Path('workflow/scripts'))/name for name in ('affinity_study_common.py','prepare_affinity_study.py',
            'affinity_study_worker.py','run_affinity_study.py','report_affinity_study.py','antigen_diagnostics.py',
            'train_antigen_diagnostic.py','model_common.py','iglm_baseline.py','chain_examples.py',
            'asd_transfer_embeddings.py','prepare_asd_transfer.py','number_diagnostic_regions.py')]
        self.protocol=dict(version=1,settings=self.settings,implementation_sha256={str(p):digest(p) for p in files},
            interventions=list(INTERVENTIONS),screening_endpoint='affinity when independently feasible; otherwise heavy-chain NLL',
            confirmation_unit='held-out source ab_ag_cluster; matched seed variation reported separately',
            external_policy='Already inspected OpenAlphaSeq is exploratory and excluded from all training',
            operational_policy='Subprocess completion waits only; no metric polling; one GPU stage at a time')
        freeze(self.root/'protocol.json',self.protocol)

    def state(self,status='running'):
        write_json(self.root/'state.json',dict(status=status,pid=os.getpid(),jobs=self.jobs,
            decisions=self.decisions,failed_stages=self.failed,updated_at=time.time()))

    def stage(self,name,target):
        for path,expected in self.protocol['implementation_sha256'].items():
            if digest(path)!=expected:raise ValueError('Implementation changed during frozen study: '+path)
        if name in self.jobs:return self.jobs[name]['status']=='complete'
        if shutil.disk_usage(self.root).free<10*2**30:raise RuntimeError('Less than 10 GiB free disk; stopping cleanly')
        logs=self.root/'controller_logs';logs.mkdir(exist_ok=True)
        command=['.snakemake/gpu-driver/bin/snakemake','-s','experiments/workflow/affinity_improvement.smk',
            '--profile','workflow/profiles/gpu','--rerun-incomplete','--rerun-triggers','mtime',
            '--configfile',self.config,'--',str(target)]
        env=dict(os.environ,MANGO_AFFINITY_STUDY_CONFIG=self.config)
        for attempt in range(2):
            log=logs/f'{name}_attempt{attempt}.log'
            with log.open('w') as handle:
                # Blocking wait: logs and results are read ONLY after child exit.
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:code=process.wait()
                except BaseException:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
            if code==0:
                self.jobs[name]=dict(status='complete',log=str(log));self.state();return True
            text=log.read_text()
            # Snakemake includes failed job logs via the GPU profile. A scalar
            # one-example microbatch cannot shrink further on OOM.
            failure=retry_kind(text)
            if attempt==0 and failure=='transient':continue
            self.jobs[name]=dict(status='failed',log=str(log),reason=failure or 'stage_error',
                                 minimum_microbatch=1 if failure=='oom' else None)
            if name not in self.failed:self.failed.append(name)
            self.state();return False

    def spec(self,phase,arm,seed,kind='mango',changes=(),source=None,shuffled=False,ce_reference=None):
        name=f'{phase}_{arm}_s{seed}'
        value=dict(id=name,phase=phase,arm=arm,seed=seed,kind=kind,changes=list(changes),source=source,
            shuffled=shuffled,ce_reference=ce_reference,
            epochs=self.settings['pilot_epochs'] if phase=='pilot' else self.settings['confirmation_epochs'])
        freeze(self.root/'specs'/f'{name}.json',value)
        if self.stage(name,self.root/'runs'/name/'complete.json'):return name
        return None

    def result(self,name):
        path=self.root/'runs'/name/'complete.json'
        return read_json(path) if path.exists() else None

    def baseline(self,phase,seed,kind='mango'):
        return self.spec(phase,'base' if kind=='mango' else 'iglm',seed,kind)

    def candidate(self,phase,arm,seed):
        if arm=='decoder':
            source=self.baseline(phase,seed,'iglm')
            if not source:return None,None
            continuation=self.spec(phase,'iglm_continue',seed,'iglm',source=source)
            candidate=self.spec(phase,'decoder',seed,'iglm_cross',source=source)
            # The paired/shuffled branches share the exact base checkpoint.
            self.spec(phase,'decoder_shuffled',seed,'iglm_cross',source=source,shuffled=True)
            return candidate,continuation
        base=self.baseline(phase,seed)
        reference=self.baseline(phase,seed,'iglm')
        if not base or not reference:return None,None
        self.spec(phase,'iglm_'+arm,seed,'iglm',changes=[arm],ce_reference=reference)
        return self.spec(phase,arm,seed,changes=[arm],ce_reference=base),base

    def screen(self,arm,audit):
        if arm=='ranking' and not audit['independent_affinity']['feasible']:
            return dict(advance=False,status='data_infeasible',reason=audit['independent_affinity']['reason'])
        if arm=='diversity' and not audit['variants']['pilot']['diversity_feasible']:
            return dict(advance=False,status='data_infeasible',reason='no_diversity_increase_under_exact_token_matching')
        differences=[];affinity=audit['independent_affinity']['feasible'];pairs=[]
        for seed in self.settings['pilot_seeds']:
            candidate,base=self.candidate('pilot',arm,seed)
            if not candidate or not base:continue
            a,b=self.result(candidate),self.result(base)
            if affinity:
                if a['development_affinity'] is None or b['development_affinity'] is None:continue
                value=a['development_affinity']-b['development_affinity']
                if a['development_heavy_nll']>b['development_heavy_nll']+self.settings['nll_tolerance']:value=-1.
            else:value=b['development_heavy_nll']-a['development_heavy_nll']
            differences.append(value);pairs.append(dict(candidate=candidate,baseline=base))
        decision=advancement(differences,self.settings,audit['development_groups'].get('val',0),affinity)
        return dict(decision,status='screened',pairs=pairs,endpoint='affinity' if affinity else 'sequence_modeling_only')

    def compare(self,pairs,audit):
        # Confirmation uses upstream test as a previously unused development
        # confirmation panel. External benchmark remains untouched until selection.
        if len(pairs)!=len(self.settings['confirmation_seeds']):
            return dict(confirmed=False,pvalue=1.,reason='missing_confirmation_seeds')
        groups=defaultdict(lambda:[0.,0]);seed_differences=[]
        affinity=audit['independent_affinity']['feasible']
        for candidate,base in pairs:
            a,b=self.result(candidate),self.result(base)
            seed_differences.append(b['development_heavy_nll']-a['development_heavy_nll'])
            left={(r['id'],r['chain'],r['variant']):r for r in read_csv(self.root/'runs'/candidate/'scores.csv')}
            right={(r['id'],r['chain'],r['variant']):r for r in read_csv(self.root/'runs'/base/'scores.csv')}
            if left.keys()!=right.keys():raise ValueError('Confirmation target membership differs')
            for key,row in left.items():
                if key[1:]!=('heavy','correct'):continue
                other=right[key]
                if row['ntokens']!=other['ntokens'] or row['group']!=other['group']:raise ValueError('Unpaired confirmation data')
                groups[row['group']][0]+=float(other['loss_sum'])-float(row['loss_sum'])
                groups[row['group']][1]+=int(row['ntokens'])
        data=[dict(improvement_sum=v[0],ntokens=v[1]) for v in groups.values()]
        boot=paired_group_bootstrap(data,self.settings['bootstrap_repeats'])
        # Affinity selection must be confirmed with an affinity-specific test;
        # do not substitute NLL significance for an affinity claim.
        if affinity:
            from report_affinity_study import affinity_confirmation
            boot=affinity_confirmation(self.root,pairs,self.settings)
        sufficient=len(groups)>=self.settings['minimum_development_groups']
        acceptable_nll=all(self.result(a)['development_heavy_nll']<=self.result(b)['development_heavy_nll']+
                           self.settings['nll_tolerance'] for a,b in pairs)
        threshold=self.settings['affinity_improvement'] if affinity else self.settings['nll_improvement']
        confirmed=sufficient and acceptable_nll and boot.get('low') is not None and boot['low']>0 and boot['improvement']>=threshold
        return dict(boot,confirmed=confirmed,seed_nll_improvements=seed_differences,
                    seed_nll_range=[min(seed_differences),max(seed_differences)],
                    endpoint='affinity' if affinity else 'sequence_modeling_only',pairs=pairs)

    def run(self):
        self.state()
        if not self.settings.get('smoke'):self.stage('latest_probes',self.root/'probes/complete.json')
        if not self.stage('audit',self.root/'cohort_audit.json'):return self.finish()
        audit=read_json(self.root/'cohort_audit.json')
        if not audit['training_feasible']:
            self.decisions['study']=dict(status='data_infeasible',reason='insufficient_protected_training_or_development_records')
            return self.finish()
        if not self.stage('embeddings',self.root/'embeddings_complete.json'):return self.finish()
        if not self.stage('regions',self.root/'regions.json'):return self.finish()
        for seed in self.settings['pilot_seeds']:
            self.baseline('pilot',seed);self.baseline('pilot',seed,'iglm')
        for arm in INTERVENTIONS:
            self.decisions[arm]=self.screen(arm,audit);self.state()
        confirmations={}
        for arm in INTERVENTIONS:
            if not self.decisions[arm]['advance']:continue
            if arm=='diversity' and not audit['variants']['confirmation']['diversity_feasible']:continue
            pairs=[]
            for seed in self.settings['confirmation_seeds']:
                candidate,base=self.candidate('confirmation',arm,seed)
                if candidate and base:
                    pairs.append((candidate,base))
                    if arm!='decoder':self.spec('confirmation',arm+'_shuffled',seed,changes=[arm],shuffled=True,ce_reference=base)
            confirmations[arm]=self.compare(pairs,audit)
        corrected=holm({arm:confirmations.get(arm,{}).get('pvalue',1.) for arm in INTERVENTIONS},
                       alpha=self.settings['confirmation_alpha'])
        confirmed=[a for a in INTERVENTIONS if confirmations.get(a,{}).get('confirmed') and corrected[a]]
        self.decisions['confirmation']=dict(results=confirmations,holm_rejected=corrected,confirmed=confirmed)
        # Combinations exclude decoder-only warm starts: initialization/freeze
        # changes require the same-budget IgLM continuation anchor at every step.
        # The generic chain below supports that anchor explicitly.
        accepted=[];previous={};combined=[]
        for arm in confirmed:
            proposed=accepted+[arm];pairs=[];new={}
            for seed in self.settings['confirmation_seeds']:
                base=previous.get(seed) or self.baseline('confirmation',seed)
                source=None;kind='mango'
                if 'decoder' in proposed:
                    source=self.baseline('confirmation',seed,'iglm');kind='iglm_cross'
                    if not previous.get(seed):base=self.spec('confirmation','iglm_continue',seed,'iglm',source=source)
                name=self.spec('confirmation','combined_'+'_'.join(proposed),seed,kind,changes=proposed,
                               source=source,ce_reference=base)
                if name and base:pairs.append((name,base));new[seed]=name
            result=self.compare(pairs,audit);combined.append(dict(changes=proposed,result=result))
            # Each extra addition uses a conservative familywise threshold.
            if result['confirmed'] and result['pvalue']<=self.settings['combination_alpha']/(2**len(INTERVENTIONS)-1):
                accepted=proposed;previous=new
        self.decisions['combinations']=combined
        external_runs=[]
        if accepted and audit['variants']['large']['records']>audit['variants']['confirmation']['records']:
            pairs=[]
            for seed in self.settings['confirmation_seeds']:
                base=self.baseline('large',seed);iglm=self.baseline('large',seed,'iglm')
                kind='iglm_cross' if 'decoder' in accepted else 'mango'
                source=iglm if kind=='iglm_cross' else None
                if source:base=self.spec('large','iglm_continue',seed,'iglm',source=source)
                candidate=self.spec('large','combined',seed,kind,changes=accepted,source=source,ce_reference=base)
                if candidate and base:pairs.append((candidate,base));external_runs.extend([candidate,base,iglm])
            self.decisions['large']=self.compare(pairs,audit)
        elif previous:
            external_runs=list(previous.values())+[f'confirmation_iglm_s{s}' for s in self.settings['confirmation_seeds']]
        else:
            external_runs=[f'pilot_{arm}_s{s}' for s in self.settings['pilot_seeds'] for arm in ('base','iglm')]
        external_runs=[] if self.settings.get('smoke') else sorted({r for r in external_runs if r and self.result(r)})
        freeze(self.root/'selection.json',dict(external_runs=external_runs,accepted_changes=accepted,
            decisions=self.decisions,selection_uses_external_labels=False))
        (self.root/'external').mkdir(exist_ok=True)
        if external_runs:self.stage('external',self.root/'external/complete.json')
        return self.finish()

    def finish(self):
        self.state('reporting')
        self.stage('report',self.root/'results_report.json')
        self.state('incomplete' if self.failed else 'complete')
        return int(bool(self.failed))


def main(config,smoke=False):
    import yaml
    def terminate(signum,frame):raise InterruptedError('Study controller terminated')
    signal.signal(signal.SIGTERM,terminate)
    settings=yaml.safe_load(Path(config).read_text())
    if smoke:
        settings.update(experiment_root='artifacts/experiments/affinity_improvement_smoke_v1',smoke=True,
                        pilot_seeds=[0],pilot_epochs=1,confirmation_seeds=[10],bootstrap_repeats=20)
        root=Path(settings['experiment_root']);root.mkdir(parents=True,exist_ok=True)
        config=str(root/'settings.yaml')
        content=yaml.safe_dump(settings,sort_keys=False)
        if Path(config).exists() and Path(config).read_text()!=content:raise ValueError('Smoke configuration changed')
        Path(config).write_text(content)
    root=Path(settings['experiment_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller(config)
        try:return controller.run()
        except Exception as error:
            write_json(root/'fatal.json',dict(error=repr(error),time=time.time()))
            controller.failed.append('controller');controller.finish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='experiments/config/affinity_improvement_study.yaml')
    p.add_argument('--smoke',action='store_true')
    raise SystemExit(main(**vars(p.parse_args())))
