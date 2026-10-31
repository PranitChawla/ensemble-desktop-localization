"""Offline paired analysis of retained study predictions."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired_comparison(baseline, candidate, *, repeats=5000, seed=20261031):
    base={row['id']:row for row in baseline}
    cand={row['id']:row for row in candidate}
    ids=sorted(base.keys() & cand.keys())
    if not ids:
        raise ValueError('No matched examples')
    clusters=defaultdict(list)
    wins=losses=0
    for identity in ids:
        a,b=int(base[identity]['correct']),int(cand[identity]['correct'])
        clusters[base[identity].get('image',identity)].append(b-a)
        wins+=b>a
        losses+=a>b
    sums=np.array([sum(values) for values in clusters.values()],dtype=float)
    sizes=np.array([len(values) for values in clusters.values()],dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(clusters),size=(repeats,len(clusters)))
    effects=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    discordant=wins+losses
    p=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/(2**discordant)) if discordant else 1.0
    return {'n':len(ids),'screenshots':len(clusters),
            'baseline_accuracy':sum(base[i]['correct'] for i in ids)/len(ids),
            'candidate_accuracy':sum(cand[i]['correct'] for i in ids)/len(ids),
            'delta':(wins-losses)/len(ids),'cluster_bootstrap_95ci':np.quantile(effects,[0.025,0.975]).tolist(),
            'candidate_wins':wins,'candidate_losses':losses,'mcnemar_exact_two_sided_p':p,
            'bootstrap_repeats':repeats,'seed':seed}


def oracle_union(predictions):
    by_model={alias:{r['id']:r for r in rows} for alias,rows in predictions.items()}
    ids=sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    union=sum(any(rows[i]['correct'] for rows in by_model.values()) for i in ids)
    qwen=by_model.get('qwen',{})
    rescue={alias:sum(not qwen[i]['correct'] and rows[i]['correct'] for i in ids)
            for alias,rows in by_model.items() if alias!='qwen'} if qwen else {}
    return {'n':len(ids),'oracle_correct':union,'oracle_accuracy':union/len(ids) if ids else None,
            'qwen_misses_recovered_by_model':rescue,
            'warning':'Diagnostic upper bound; target labels are unavailable to actual policies'}


def summarize_directory(root):
    root=Path(root)
    summaries={}
    for path in sorted(root.glob('worker-*/*/summary.json')):
        summary=json.loads(path.read_text())
        summaries[summary['name']]=summary
    return summaries



def sampling_diagnostics(rows):
    """Offline label-based coverage diagnostics; never available to inference policies."""
    from grounding.evaluation import score_point

    consensus_rows=[]
    prefix_hits=defaultdict(int)
    by_model=defaultdict(lambda:{'proposals':0,'correct':0,'valid':0})
    all_correct=valid=total=0
    for row in rows:
        prediction=row.get('prediction',{})
        candidates=sorted(prediction.get('candidates',{}).items(),key=lambda item:int(item[0].rsplit('-',1)[1]))
        outcomes=[]
        for alias,point in candidates:
            hit=score_point(point,row['bbox_xyxy'],row['width'],row['height'])
            outcomes.append(hit)
            model=alias.rsplit('-',1)[0]
            by_model[model]['proposals']+=1
            by_model[model]['valid']+=point is not None
            by_model[model]['correct']+=hit
            total+=1
            valid+=point is not None
        for count in range(1,6):
            prefix_hits[count]+=any(hit for (alias,_),hit in zip(candidates,outcomes)
                                    if int(alias.rsplit('-',1)[1])<count)
        all_correct+=any(outcomes)
        point=prediction.get('consensus_point')
        consensus_rows.append(dict(row,correct=score_point(point,row['bbox_xyxy'],row['width'],row['height']),
                                   prediction={'point':point}))
    n=len(rows)
    return {'n':n,'selector_correct':sum(r['correct'] for r in rows),
            'consensus_correct':sum(r['correct'] for r in consensus_rows),
            'oracle_any_correct':all_correct,'oracle_accuracy':all_correct/n if n else None,
            'oracle_prefix_correct':dict(prefix_hits),'proposal_count':total,'valid_proposals':valid,
            'by_model':dict(by_model),'consensus_rows':consensus_rows,
            'warning':'Oracle coverage uses labels offline; it is not achieved inference accuracy'}
