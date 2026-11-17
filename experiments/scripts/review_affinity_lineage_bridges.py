"""Independent graph traversal and schema-aware review of cached CDR3 labels.

Region caches contain a trailing EOS label for autoregressive scoring. It must
be removed explicitly before matching labels to residues. This supplement does
not modify the frozen parent audit or assert that different CDR3s are clones
from independent parents.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import re

from affinity_study_common import digest, fingerprint, freeze, read_csv, read_json, write_csv, write_json
from audit_affinity_bottleneck import edit_distance


def cdr3_labels(item):
    sequence=item['sequence'];labels=list(item['labels'])
    if len(labels)==len(sequence)+1 and labels[-1]=='eos':
        labels=labels[:-1]
    if len(labels)!=len(sequence):
        raise ValueError('Region labels do not align to residues, including explicit EOS handling')
    return ''.join(aa for aa,label in zip(sequence,labels) if label=='cdr3')


def traverse_components(adjacency, row_nodes):
    """BFS independent of the union-find implementation used in the main audit."""
    remaining=set(row_nodes);parts=[]
    while remaining:
        start=min(remaining);seen={start};queue=deque([start]);members=set()
        while queue:
            node=queue.popleft()
            if node in row_nodes:members.add(node)
            for nxt in adjacency[node]:
                if nxt not in seen:seen.add(nxt);queue.append(nxt)
        remaining-=members;parts.append(tuple(sorted(n[4:] for n in members)))
    return sorted(parts,key=lambda p:(-len(p),p))


def main():
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='experiments/config/affinity_lineage_review.yaml')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg['experiment_root']);root.mkdir(parents=True,exist_ok=True);audit=Path(cfg['audit_root'])
    inputs=[args.config,cfg['source_records'],cfg['candidate_links'],*cfg['region_caches'],
            str(audit/'results_report.json'),str(audit/'typed_graph_edges.csv'),
            str(audit/'antigen_exclusion_witnesses.csv'),str(audit/'native_affinity_deduplicated.csv')]
    freeze(root/'protocol.json',dict(inputs_sha256={p:digest(p) for p in inputs},
           implementation_sha256={str(Path(__file__)):digest(__file__),
                                  'experiments/scripts/audit_affinity_bottleneck.py':digest('experiments/scripts/audit_affinity_bottleneck.py')},
           scope='Independent graph verification and supplementary CDR3 diagnostics only.'))
    report=read_json(audit/'results_report.json')
    for name,value in report['output_sha256'].items():
        if digest(audit/name)!=value:raise ValueError('Main audit output changed: '+name)
    edges=read_csv(audit/'typed_graph_edges.csv');adjacent=defaultdict(set);typed=set();row_nodes=set()
    for r in edges:
        a,b,k=r['a'],r['b'],r['edge_type'];adjacent[a].add(b);adjacent[b].add(a)
        typed.add((a,b,k));typed.add((b,a,k))
        if a.startswith('row:'):row_nodes.add(a)
        if b.startswith('row:'):row_nodes.add(b)
    parts=traverse_components(adjacent,row_nodes)
    if len(parts)!=report['graph']['components'] or len(parts[0])!=report['graph']['largest_component']:
        raise ValueError('Independent BFS disagrees with the full graph')
    witnesses=read_csv(audit/'antigen_exclusion_witnesses.csv')
    direct={r['id'] for r in witnesses if r['direct']=='True'}
    for r in witnesses:
        nodes=json.loads(r['path_nodes']);kinds=r['edge_types'].split(';') if r['edge_types'] else []
        if len(nodes)!=len(kinds)+1 or nodes[0]!='row:'+r['witness_id'] or nodes[-1]!='row:'+r['id']:
            raise ValueError('Malformed propagation witness')
        if r['witness_id'] not in direct or any((a,b,k) not in typed for a,b,k in zip(nodes,nodes[1:],kinds)):
            raise ValueError('Propagation witness is not supported by the typed graph')
    candidates=read_json(cfg['candidate_links'])
    numeric=[r for r in candidates if re.fullmatch(r'[1-9][0-9]*',r['pmid'])]
    if len(numeric)!=report['native']['numeric_publication_links'] or len({r['pmid'] for r in numeric})!=report['native']['numeric_publications']:
        raise ValueError('Independent publication census disagrees')
    unique=read_csv(audit/'native_affinity_deduplicated.csv')
    if sum(int(r['copy_count']) for r in unique)!=len(candidates):
        raise ValueError('Construct-copy accounting failed')
    known={}
    for path in cfg['region_caches']:
        for key,item in read_json(path)['sequences'].items():
            if fingerprint(item['sequence'])!=key:raise ValueError('Region-cache sequence hash changed')
            if item['status']!='numbered':continue
            value=cdr3_labels(item)
            if key in known and known[key]!=value:raise ValueError('Cached CDR3 annotations disagree')
            if value:known[key]=value
    records={r['id']:r for r in read_csv(cfg['source_records'])};roles=defaultdict(set);groups=defaultdict(set)
    for node in row_nodes:
        row=records[node[4:]]
        for c in ('H','L'):
            seq=row[f'resolved_{c}_seq']
            if seq:
                key=fingerprint(seq);roles[key].add(c);groups[key].add(row['ab_ag_cluster'])
    annotated=[]
    for edge in edges:
        if edge['edge_type']!='near_antibody':continue
        a=edge['a'].split(':')[1];b=edge['b'].split(':')[1];av=known.get(a,'');bv=known.get(b,'')
        annotated.append(dict(a=a,b=b,roles_a=';'.join(sorted(roles[a])),roles_b=';'.join(sorted(roles[b])),
                              both_cdr3_numbered=bool(av and bv),cdr3_a=av,cdr3_b=bv,
                              cdr3_identical=av==bv if av and bv else '',
                              cdr3_edit_distance=edit_distance(av,bv) if av and bv else '',
                              connects_disjoint_source_groups=not bool(groups[a]&groups[b])))
    write_csv(root/'near_edges.csv',annotated)
    numbered=[r for r in annotated if r['both_cdr3_numbered']]
    results=dict(independent_bfs_components=len(parts),largest_component=len(parts[0]),
                 validated_propagation_witnesses=len(witnesses),
                 numeric_publication_links=len(numeric),numeric_publications=len({r['pmid'] for r in numeric}),
                 total_near_antibody_edges=len(annotated),both_cdr3_numbered=len(numbered),
                 numbered_edges_with_different_cdr3=sum(r['cdr3_identical'] is False for r in numbered),
                 numbered_edges_across_source_groups=sum(r['connects_disjoint_source_groups'] for r in numbered),
                 new_independent_lineages_asserted=False,grouping_changes_applied=False,
                 caveats=['CDR3 label coverage is incomplete and nonrandom.',
                          'Different CDR3s can occur within a lineage; similar CDR3s can be convergent.',
                          'A revised lineage rule needs V/J, CDR3 length/identity and experimental parent metadata.',
                          'No graph diagnostic authorizes a change to prior holdouts or group-aware splits.'])
    write_json(root/'results.json',results);print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
