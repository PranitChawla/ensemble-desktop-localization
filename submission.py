"""Editable Python inference program. The optimization agent changes this file.

The fixed VLMs return grounding predictions; they never generate or execute Python.
The initial policy is a single native-protocol query, pending development selection.
"""
from __future__ import annotations

from PIL import Image

from grounding.coordinates import ImageFrame


def ground(screenshot, instruction, pool, *, controller='venus'):
    alias=controller if controller in pool.models else next(iter(pool.models))
    with Image.open(screenshot) as image:
        frame=ImageFrame.full(image.size)
    result=pool.ground(alias,instruction,image=screenshot,frame=frame)
    return {'status':'completed','point':result['point_original_norm01'],
            'calls':pool.calls,'protocol':result}
