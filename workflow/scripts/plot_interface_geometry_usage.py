"""Show the fixed geometry-scrambling intervention on the trained contact head."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from affinity_study_common import read_json


def main(root):
    root=Path(root);r=read_json(root/'results_report.json');fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    labels=[]
    for i,(arm,variant,label) in enumerate([
        ('geom_ce','correct','3D + CE/probe / correct'),('geom_ce','scrambled','3D + CE/probe / permuted'),
        ('geom_contact','correct','3D + contact gradients / correct'),('geom_contact','scrambled','3D + contact gradients / permuted')]):
        x=next(x for x in r['summary'] if (x['arm'],x['variant'],x['metric'])==(arm,variant,'contact_ap'))
        ax.errorbar(x['mean'],i,xerr=x['seed_sd'],fmt='o',capsize=4)
        ax.scatter(x['seed_points'],np.full(len(x['seed_points']),i),s=15,alpha=.6);labels.append(label)
    ax.set(yticks=range(4),yticklabels=labels,xlabel='Held-out within-residue contact AP',
           title='Contact learning makes the predictor use geometry\nBars: ±1 optimization-seed SD')
    for suffix in ('png','svg'):fig.savefig(root/f'geometry_usage.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/experiments/interface_supervision_v1');a=p.parse_args();main(a.root)
