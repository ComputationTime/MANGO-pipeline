"""Analysis helpers; structural contacts are output annotations, never inputs."""
import math
import numpy as np

REGIONS=('full','framework','cdr1','cdr2','cdr3','cdrs')


def pooling_weights(length,tokens=32):
    weights=np.zeros((tokens,length),dtype=float)
    for i in range(tokens):
        start=math.floor(i*length/tokens);end=math.ceil((i+1)*length/tokens)
        weights[i,start:end]=1/(end-start)
    return weights


def masks(labels):
    return dict(full=np.ones(len(labels),bool),framework=np.array([x=='framework' for x in labels]),
                cdr1=np.array([x=='cdr1' for x in labels]),cdr2=np.array([x=='cdr2' for x in labels]),
                cdr3=np.array([x=='cdr3' for x in labels]),cdrs=np.array([x.startswith('cdr') for x in labels]))


def weighted_difference(values,fraction):
    values=np.asarray(values,float);fraction=np.asarray(fraction,float)
    if fraction.sum()<=0 or (1-fraction).sum()<=0:return None
    return float(np.dot(values,fraction)/fraction.sum()-np.dot(values,1-fraction)/(1-fraction).sum())


def safe_rho(a,b):
    from scipy.stats import spearmanr
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).statistic)


def contact_annotations(row,cutoffs):
    """Map observed residues exactly; all coordinates stay in this analysis step."""
    from Bio.PDB import MMCIFParser,NeighborSearch
    from Bio.Data.PDBData import protein_letters_3to1_extended as aa
    model=MMCIFParser(QUIET=True).get_structure(row['id'],row['pdb_path'])[0]
    def residues(chain,sequence):
        observed=[r for r in model[chain] if r.get_resname() in aa and r.id[0]!='W']
        string=''.join(aa[r.get_resname()] for r in observed)
        start=string.find(sequence)
        if start<0 or string.find(sequence,start+1)>=0:raise ValueError('Observed structure does not uniquely match target sequence: '+chain)
        return observed[start:start+len(sequence)]
    heavy=residues(row['heavy_chain'],row['resolved_H_seq'])
    def atoms(rs):
        return [(a,i) for i,r in enumerate(rs) for a in r.get_atoms() if a.element not in ('H','D')]
    h_atoms=atoms(heavy);owners={id(a):i for a,i in h_atoms};tree=NeighborSearch([a for a,i in h_atoms])
    hdist=np.full(len(heavy),9.0);agdist=[];mapping=[]
    chains=row['antigen_chains'].split(',');sequences=row['resolved_ag_seq'].split(',')
    for chain,sequence in zip(chains,sequences):
        rs=residues(chain,sequence);dist=np.full(len(rs),9.0)
        for atom,i in atoms(rs):
            for other in tree.search(atom.coord,8.0,level='A'):
                d=float(np.linalg.norm(atom.coord-other.coord));j=owners[id(other)]
                dist[i]=min(dist[i],d);hdist[j]=min(hdist[j],d)
        if agdist:
            agdist.append(9.0);mapping.append(dict(chain_break=True))
        agdist.extend(dist.tolist());mapping.extend(dict(chain=chain,index=i) for i in range(len(rs)))
    result=dict(id=row['id'],heavy_length=len(heavy),embedding_length=len(agdist),
                heavy_distances_capped_at_9A=hdist.tolist(),antigen_distances_capped_at_9A=agdist,
                antigen_residue_mapping=mapping,contacts={})
    for cutoff in cutoffs:
        result['contacts'][str(float(cutoff))]=dict(
            paratope=np.flatnonzero(hdist<=cutoff).tolist(),
            epitope=np.flatnonzero(np.asarray(agdist)<=cutoff).tolist())
    return result
