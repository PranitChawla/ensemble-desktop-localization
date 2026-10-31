"""Editable Python inference program. The optimization agent changes this file.

The fixed VLMs return grounding predictions; they never generate or execute Python.
Development-selected policy: one native Venus query at four megapixels.
"""
from __future__ import annotations

from PIL import Image

from grounding.coordinates import ImageFrame


def ground(screenshot, instruction, pool, *, controller='venus'):
    alias=controller if controller in pool.models else next(iter(pool.models))
    with Image.open(screenshot) as image:
        frame=ImageFrame.full(image.size)
    result=pool.ground(alias,instruction,image=screenshot,frame=frame,
                       processor_kwargs={'max_pixels':4194304} if alias=='venus' else None)
    return {'status':'completed','point':result['point_original_norm01'],
            'calls':pool.calls,'protocol':result}
