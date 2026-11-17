"""Export a compact scientific summary of frozen diagnostic results."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from affinity_study_common import read_json


def main(root):
    root=Path(root);r=read_json(root/'results_report.json');v=read_json(root/'independent_validation.json')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    items=[]
    for condition,region,label in [('correct','full','Full chain / correct antigen'),('mean','full','Full chain / averaged antigen'),
            ('none','full','Full chain / antigen absent'),('correct','framework','Framework / correct antigen'),('correct','cdrs','All CDRs / correct antigen')]:
        values=[x for x in r['correlations'] if (x['arm'],x['condition'],x['region'])==('MANGO_CE',condition,region)]
        points=[statistics.mean(x['rho']['seed_points'][i] for x in values) for i in range(5)]
        items.append((label,points))
    ax=axes[0,0]
    for i,(label,points) in enumerate(items):
        ax.errorbar(statistics.mean(points),i,xerr=statistics.stdev(points),fmt='o',capsize=4)
        ax.scatter(points,np.full(len(points),i),s=12,alpha=.55)
    ax.set(yticks=range(len(items)),yticklabels=[x[0] for x in items],xlabel='Macro affinity Spearman rho',title='A. Antigen detail adds little to affinity ranking')
    ax=axes[0,1]
    for arm,color,offset in [('MANGO_CE','#356f9e',-.08),('MANGO_rank','#c66d2b',.08)]:
        for i,region in enumerate(['full','cdr3','paratope']):
            x=next(x for x in r['structural'] if x['arm']==arm and x['region']==region and x['cutoff']==5.)['attention_enrichment']
            ax.errorbar(x['mean'],i+offset,xerr=x['seed_sd'],fmt='o',capsize=4,color=color,label=arm if i==0 else None)
    ax.axvline(1,color='gray',ls='--');ax.legend(fontsize=8)
    ax.set(yticks=range(3),yticklabels=['All predictions','CDR3 predictions','Paratope predictions'],xlabel='Epitope enrichment (uniform = 1)',title='B. Attention shows little epitope concentration')
    ax=axes[1,0]
    for arm,color,offset in [('MANGO_CE','#356f9e',-.08),('MANGO_rank','#c66d2b',.08)]:
        for i,region in enumerate(['full','cdr3','paratope']):
            x=next(x for x in r['structural'] if x['arm']==arm and x['region']==region and x['cutoff']==5.)['occlusion_epitope_rho']
            ax.errorbar(x['mean'],i+offset,xerr=x['seed_sd'],fmt='o',capsize=4,color=color)
    ax.axvline(0,color='gray',ls='--')
    ax.set(yticks=range(3),yticklabels=['All predictions','CDR3 predictions','Paratope predictions'],
           xlabel='Spearman rho (token sensitivity, epitope fraction)',title='C. Fixed token occlusions weakly align with contacts')
    ax=axes[1,1]
    for i,region in enumerate(['framework','cdr1','cdr2','cdr3']):
        x=next(x for x in v['ranking_gain_decomposition'] if x['family']=='MANGO' and x['region']==region)
        ax.errorbar(x['mean'],i,xerr=x['seed_sd'],fmt='o',capsize=4)
    ax.axvline(0,color='gray',ls='--')
    ax.set(yticks=range(4),yticklabels=['Framework','CDR1','CDR2','CDR3'],
           xlabel='Contribution to whole-chain preference gain (nats/token)',title='D. RNase-A ranking gain concentrates in CDR1')
    fig.suptitle('Epitope/paratope diagnostics — bars: ±1 optimization-seed SD',fontsize=14)
    for suffix in ('png','svg'):fig.savefig(root/f'mechanism_summary.{suffix}',dpi=190)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/experiments/epitope_diagnostics_v1');a=p.parse_args();main(a.root)
