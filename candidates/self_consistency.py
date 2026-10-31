"""Five stochastic proposals, coordinate medoid, and an independent model selection call."""
from __future__ import annotations

from grounding import study_policies as policies

SAMPLERS = ['venus'] * 5
TEMPERATURE = 0.7
TOP_P = 0.9


def consensus(points):
    if not points:
        return None
    # Select an observed point; averaging unrelated controls can create an invalid click.
    def rank(point):
        distances=[((point[0]-other[0])**2+(point[1]-other[1])**2)**0.5 for other in points]
        return (-sum(d <= 0.03 for d in distances),sum(distances))
    return min(points,key=rank)


def predict(ctx, baseline):
    pool=ctx['pool']
    candidates={}
    failures=[]
    pool.temperature,pool.top_p=TEMPERATURE,TOP_P
    try:
        for index,alias in enumerate(SAMPLERS):
            try:
                candidates[f'{alias}-{index}']=policies.direct(ctx,alias)
            except Exception as error:
                candidates[f'{alias}-{index}']={'point':None}
                failures.append({'sample_index':index,'model':alias,'error':f'{type(error).__name__}: {error}'})
    finally:
        pool.temperature,pool.top_p=0.0,1.0
    points=[value['point'] for value in candidates.values() if value.get('point') is not None]
    medoid=consensus(points)
    try:
        selected=policies.select_candidate(ctx,candidates,judge='mai',annotated=True)
    except Exception as error:
        selected={'point':medoid,'selection_error':f'{type(error).__name__}: {error}'}
    return dict(selected,candidates={alias:value['point'] for alias,value in candidates.items()},
                sampling_failures=failures,consensus_point=medoid,
                sampling_temperature=TEMPERATURE,sampling_top_p=TOP_P,
                python_operations=['coordinate_consensus','candidate_annotation'])
