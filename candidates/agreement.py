"""Choose an observed native prediction using cross-model coordinate agreement."""
from __future__ import annotations

PRIORITY = ['venus','mai','jedi']
RADIUS = 0.005


def choose(candidates, priority=PRIORITY, radius=RADIUS):
    valid=[alias for alias in priority if candidates.get(alias) is not None]
    if not valid:
        return None
    def rank(alias):
        point=candidates[alias]
        neighbors=sum(((point[0]-candidates[other][0])**2+(point[1]-candidates[other][1])**2)**0.5<=radius
                      for other in valid)
        return (-neighbors,priority.index(alias))
    return min(valid,key=rank)


def predict(ctx,baseline):
    candidates={}
    errors={}
    for alias in PRIORITY:
        try:
            if alias=='venus':
                from grounding.study_policies import direct
                candidates[alias]=direct(ctx,alias,request_pixels=4194304)['point']
            else:
                candidates[alias]=baseline(alias)['point']
        except Exception as error:
            candidates[alias]=None
            errors[alias]=f'{type(error).__name__}: {error}'
    selected=choose(candidates)
    return {'point':candidates[selected] if selected else None,'selected_alias':selected,
            'candidates':candidates,'model_errors':errors,
            'python_operations':['cross_model_agreement']}
