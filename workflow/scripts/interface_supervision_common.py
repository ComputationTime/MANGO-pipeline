"""Antigen-only graph construction and separately computed contact labels."""
import numpy as np

ARMS={'seq_ce':('sequence',False),'seq_contact':('sequence',True),
      'geom_ce':('geometry',False),'geom_contact':('geometry',True),'iglm_ce':('none',False)}


def average_precision(truth,score):
    """Threshold-grouped AP, including ties; undefined when there is no positive."""
    truth=np.asarray(truth,bool).ravel();score=np.asarray(score,float).ravel()
    if not truth.any():return None
    order=np.argsort(-score,kind='stable');y=truth[order];s=score[order]
    endpoints=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[endpoints];precision=tp/(endpoints+1);recall=tp/truth.sum()
    return float(np.dot(np.diff(np.r_[0.,recall]),precision))


def graph(coords,chains,positions,kind,k=16):
    n=len(coords);indices=np.zeros((n,k),dtype=np.int64);valid=np.zeros((n,k),bool)
    edges=np.zeros((n,k,18),dtype=np.float32)
    same=np.asarray(chains)[:,None]==np.asarray(chains)[None,:]
    sep=np.abs(np.asarray(positions)[:,None]-np.asarray(positions)[None,:])
    if kind=='geometry':dist=np.linalg.norm(np.asarray(coords)[:,None]-np.asarray(coords)[None,:],axis=-1)
    elif kind=='sequence':dist=np.where(same,sep*3.8,np.inf)
    else:raise ValueError('Unknown graph kind')
    np.fill_diagonal(dist,np.inf);centers=np.linspace(2.,30.,16)
    for i in range(n):
        neighbors=np.argsort(dist[i],kind='stable');neighbors=neighbors[np.isfinite(dist[i,neighbors])][:k]
        if not len(neighbors):neighbors=np.array([i])
        count=len(neighbors);indices[i]=i;indices[i,:count]=neighbors;valid[i,:count]=True
        d=np.where(np.isfinite(dist[i,neighbors]),dist[i,neighbors],0.)
        edges[i,:count,:16]=np.exp(-((d[:,None]-centers[None,:])/2.)**2)
        edges[i,:count,16]=same[i,neighbors]
        edges[i,:count,17]=np.where(same[i,neighbors],np.minimum(np.log1p(sep[i,neighbors])/np.log(33),1.),1.)
    return indices,edges,valid


def observed_residues(model,chain,sequence):
    from Bio.Data.PDBData import protein_letters_3to1_extended as aa
    residues=[r for r in model[chain] if r.get_resname() in aa and r.id[0]!='W']
    observed=''.join(aa[r.get_resname()] for r in residues);start=observed.find(sequence)
    if start<0 or observed.find(sequence,start+1)>=0:raise ValueError('Structure/sequence mapping is not unique: '+chain)
    return residues[start:start+len(sequence)]


def antigen_input(row):
    """This function never accesses antibody chains or target sequences."""
    from Bio.PDB import MMCIFParser
    model=MMCIFParser(QUIET=True).get_structure('antigen',row['pdb_path'])[0]
    coordinates=[];chains=[];positions=[];keep=[];offset=0;fallbacks=0
    for ci,(chain,seq) in enumerate(zip(row['antigen_chains'].split(','),row['resolved_ag_seq'].split(','))):
        if ci:offset+=1  # Existing ESM2 cache contains a chain-break CLS token.
        residues=observed_residues(model,chain,seq)
        for i,residue in enumerate(residues):
            if 'CA' in residue:coord=residue['CA'].coord
            else:
                atoms=[a.coord for a in residue if a.element not in ('H','D')]
                if not atoms:raise ValueError('Antigen residue has no heavy atoms')
                coord=np.mean(atoms,axis=0);fallbacks+=1
            coordinates.append(coord);chains.append(ci);positions.append(i);keep.append(offset+i)
        offset+=len(seq)
    return dict(coords=np.asarray(coordinates,dtype=np.float32),chains=np.array(chains),
                positions=np.array(positions),keep=np.array(keep,dtype=np.int64),ca_fallbacks=fallbacks)


def contact_labels(row,cutoff=5.):
    """Antibody atoms are accessed only here, to create loss/evaluation labels."""
    from Bio.PDB import MMCIFParser,NeighborSearch
    model=MMCIFParser(QUIET=True).get_structure('labels',row['pdb_path'])[0]
    ag=[]
    for chain,seq in zip(row['antigen_chains'].split(','),row['resolved_ag_seq'].split(',')):
        ag.extend(observed_residues(model,chain,seq))
    atoms=[a for residue in ag for a in residue if a.element not in ('H','D')]
    owners={id(a):i for i,residue in enumerate(ag) for a in residue if a.element not in ('H','D')};tree=NeighborSearch(atoms)
    result={}
    for chain,key,column in [('heavy','heavy_chain','resolved_H_seq'),('light','light_chain','resolved_L_seq')]:
        if not row[column]:result[chain]=np.zeros((0,len(ag)),dtype=bool);continue
        rs=observed_residues(model,row[key],row[column]);labels=np.zeros((len(rs),len(ag)),bool)
        for i,residue in enumerate(rs):
            for atom in residue:
                if atom.element in ('H','D'):continue
                for neighbor in tree.search(atom.coord,cutoff,level='A'):labels[i,owners[id(neighbor)]]=True
        result[chain]=labels
    return result
