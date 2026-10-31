"""Unbenchmarked API example: classifier -> grounders -> disagreement verifier.

The coding agent edits this Python decision tree. VLM outputs are data, never code.
"""
from __future__ import annotations

import json

from PIL import Image, ImageDraw

from grounding.coordinates import ImageFrame


def ground(screenshot,instruction,pool,*,controller=None):
    classifier='qwen' if 'qwen' in pool.models else 'mai'
    with Image.open(screenshot) as original:
        image=original.convert('RGB')
    schema={'type':'object','properties':{'kind':{'enum':['text','icon','spatial','uncertain']}},
            'required':['kind'],'additionalProperties':False}
    kind=json.loads(pool.ask_vlm(classifier,
        f'Classify the target requested by this instruction: {instruction}. '
        'Use text for a named visible label, icon for a pictorial control, spatial for a relative position, '
        'and uncertain when the category is unclear. Return only the JSON category.',
        image=image,response_schema=schema,max_tokens=64))['kind']
    frame=ImageFrame.full(image.size)
    primary=pool.ground('venus',instruction,image=screenshot,frame=frame,
                        processor_kwargs={'max_pixels':4194304})
    point=primary['point_original_norm01']
    trace={'kind':kind,'primary':primary}
    if kind!='icon' and point is not None:
        helper=pool.ground('mai',instruction,image=screenshot,frame=frame)
        other=helper['point_original_norm01']
        trace['helper']=helper
        if other is not None and sum((a-b)**2 for a,b in zip(point,other))**0.5>0.01:
            annotated=image.copy()
            draw=ImageDraw.Draw(annotated)
            for number,candidate in enumerate([point,other],1):
                x,y=candidate[0]*image.width,candidate[1]*image.height
                draw.ellipse((x-12,y-12,x+12,y+12),outline='red',width=3)
                draw.text((x+15,y),str(number),fill='red',stroke_width=1,stroke_fill='white')
            choice=json.loads(pool.ask_vlm(classifier,
                f'Instruction: {instruction}. Which marked candidate identifies the requested existing control? '
                'Return choice 1 or 2; choose 0 if the evidence is inconclusive.',image=annotated,
                response_schema={'type':'object','properties':{'choice':{'enum':[0,1,2]}},
                                 'required':['choice'],'additionalProperties':False},max_tokens=64))['choice']
            trace['verification_choice']=choice
            if choice==2:
                point=other
    return {'status':'completed','point':point,'calls':pool.calls,'decision_tree':trace}
