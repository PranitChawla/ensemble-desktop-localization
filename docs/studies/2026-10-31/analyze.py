"""Reproduce paired statistics and the sampling figure from retained predictions."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from grounding.evaluation import score_point
from grounding.study_analysis import oracle_union, paired_comparison, sampling_diagnostics
from grounding.study_runner import aggregate

ROOT=Path(__file__).resolve().parent
RUN='study-20261031T225702Z'


def derived_predictions(jobs,plan):
    jobs=dict(jobs)
    parts=[jobs.get(f'agreement-4mp-validation-part{i}',[]) for i in [0,1]]
    if any(parts):
        jobs['agreement-4mp-validation']=[row for part in parts for row in part]
    key='python-4mp-detail-validation'
    if key in jobs:
        rows=[]
        for row in jobs[key]:
            prediction=dict(row.get('prediction',{}))
            point,first=prediction.get('point'),prediction.get('first_point')
            if point is not None and first is not None:
                distance=sum((a-b)**2 for a,b in zip(point,first))**0.5
                if distance>plan['gate']:
                    point=first
            prediction['point']=point
            rows.append(dict(row,prediction=prediction,
                             correct=score_point(point,row['bbox_xyxy'],row['width'],row['height'])))
        jobs['python-4mp-gated-validation']=rows
    for name,rows in list(jobs.items()):
        if name.startswith('sampling-') and name.endswith('-validation'):
            derived=[]
            for row in rows:
                prediction=row.get('prediction',{})
                point=prediction.get('consensus_point')
                derived.append(dict(row,prediction={'point':point},
                                    correct=score_point(point,row['bbox_xyxy'],row['width'],row['height']),
                                    cost=prediction.get('consensus_cost',row['cost'])))
            jobs[name.replace('-validation','-consensus-validation')]=derived
    return jobs


def analyze(jobs,plan):
    jobs=derived_predictions(jobs,plan)
    pairs=list(plan['comparisons'])+[
        {'baseline':'direct-venus-4mp-validation','candidate':'python-4mp-gated-validation'},
        {'baseline':'python-4mp-full-validation','candidate':'python-4mp-gated-validation'},
        {'baseline':'direct-venus-4mp-sampling-control-validation','candidate':'sampling-venus-4mp-consensus-validation'},
        {'baseline':'direct-venus-4mp-sampling-control-validation','candidate':'sampling-mixed-4mp-consensus-validation'},
        {'baseline':'sampling-venus-4mp-consensus-validation','candidate':'sampling-mixed-4mp-consensus-validation'},
    ]
    comparisons={p['candidate']+' vs '+p['baseline']:paired_comparison(jobs[p['baseline']],jobs[p['candidate']])
                 for p in pairs if p['baseline'] in jobs and p['candidate'] in jobs}
    sampling={}
    for name,rows in jobs.items():
        if name.startswith('sampling-') and 'consensus' not in name:
            value=sampling_diagnostics(rows)
            value.pop('consensus_rows')
            sampling[name]=value
    pool={a:jobs[f'direct-{a}-validation'] for a in ['venus','mai','jedi'] if f'direct-{a}-validation' in jobs}
    oracles={}
    if len(pool)==3:
        oracles['2mp']=oracle_union(pool,reference='venus')
        pool['venus']=jobs['direct-venus-4mp-validation']
        oracles['4mp_venus']=oracle_union(pool,reference='venus')
    return {'run_id':RUN,'frozen_at_utc':plan['frozen_at_utc'],'metrics':{name:aggregate(rows) for name,rows in jobs.items()},
            'comparisons':comparisons,'sampling':sampling,'model_pool_oracles':oracles}


def figure(jobs,plan):
    import matplotlib.pyplot as plt

    fig,ax=plt.subplots(figsize=(7.2,4.1),layout='constrained')
    colors={'venus':'#2667a6','mixed':'#cb5929'}
    for label in ['venus','mixed']:
        rows=jobs.get(f'sampling-{label}-4mp-validation',[])
        if not rows:
            continue
        attainable=[]
        oracle=[]
        for count in range(1,6):
            hits=any_hits=0
            for row in rows:
                prediction=row.get('prediction',{})
                candidates=[point for alias,point in prediction.get('candidates',{}).items()
                            if int(alias.rsplit('-',1)[1])<count and point is not None]
                def rank(point):
                    distances=[sum((a-b)**2 for a,b in zip(point,other))**0.5 for other in candidates]
                    return (-sum(d<=plan['consensus_radius'] for d in distances),sum(distances))
                point=min(candidates,key=rank) if candidates else None
                hits+=score_point(point,row['bbox_xyxy'],row['width'],row['height'])
                any_hits+=any(score_point(p,row['bbox_xyxy'],row['width'],row['height']) for p in candidates)
            attainable.append(hits/len(rows)*100)
            oracle.append(any_hits/len(rows)*100)
        name='Venus only' if label=='venus' else 'Mixed models'
        ax.plot(range(1,6),attainable,'o-',color=colors[label],label=name+' · consensus')
        ax.plot(range(1,6),oracle,'o--',color=colors[label],alpha=.75,label=name+' · oracle coverage')
    control=jobs.get('direct-venus-4mp-sampling-control-validation',[])
    if control:
        accuracy=sum(r['correct'] for r in control)/len(control)*100
        ax.axhline(accuracy,color='#555555',linestyle=':',label='One greedy Venus call')
    ax.set(xlabel='Stochastic proposal calls',ylabel='Point-in-box accuracy (%)',xticks=range(1,6),
           title='Repeated sampling: usable consensus vs candidate coverage')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=8,loc='best',frameon=False)
    fig.savefig(ROOT/'sampling.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    with gzip.open(ROOT/(RUN+'-predictions.json.gz'),'rt') as stream:
        jobs=json.load(stream)
    plan=json.loads((ROOT/'frozen-plan.json').read_text())
    result=analyze(jobs,plan)
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    figure(jobs,plan)
    print(ROOT/'analysis.json')
