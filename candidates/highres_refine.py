"""Two-call four-megapixel policy; Python can inspect a native-coordinate detail view."""
from __future__ import annotations

from grounding import study_policies as policies

MODE = 'detail'
FRACTION = 0.3
MAX_DISPLACEMENT = None


def predict(ctx,baseline):
    pool=ctx['pool']
    original=pool.chat
    def chat(alias,messages,**kwargs):
        if alias=='venus' and kwargs.get('processor_kwargs') is None:
            kwargs['processor_kwargs']={'max_pixels':4194304}
        return original(alias,messages,**kwargs)
    pool.chat=chat
    first=policies.direct(ctx,'venus')
    result=policies.repeat_refine(ctx,first,alias='venus',mode=MODE,fraction=FRACTION,
                                  fallback=True,max_displacement=MAX_DISPLACEMENT)
    result['first_point']=first['point']
    return result
