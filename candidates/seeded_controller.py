"""Development treatment: a controller verifies a first-pass estimate with optional Python."""
from __future__ import annotations

import json

from grounding.client import image_url
from grounding.coordinates import ImageFrame, to_original_point
from grounding.tools import PythonSession

PYTHON_ENABLED = True


def predict(ctx, baseline):
    first = baseline('qwen')
    pool = ctx['pool']
    actions = [
        {'type':'object','properties':{'action':{'const':'final'},'point':{'type':'array',
         'items':{'type':'number'},'minItems':2,'maxItems':2}},
         'required':['action','point'],'additionalProperties':False},
        {'type':'object','properties':{'action':{'const':'model'},'model':{'enum':['qwen','ministral','lfm']}},
         'required':['action','model'],'additionalProperties':False},
    ]
    if PYTHON_ENABLED:
        actions.append({'type':'object','properties':{'action':{'const':'python'},'code':{'type':'string'}},
                        'required':['action','code'],'additionalProperties':False})
    system = (
        'Your ONLY task is to locate an existing interface control in the screenshot. '
        'The user instruction describes which control to find. Do not carry out the instruction, '
        'make a plot, write an application, or invent a screenshot. '
        'A first estimate is supplied. Keep it unless evidence supports a correction. '
        'Return JSON actions. Final coordinates MUST be 0-1000 relative to the ORIGINAL screenshot: '
        '{"action":"final","point":[500,500]}. '
        'A helper query is {"action":"model","model":"qwen"}; helpers qwen, ministral, lfm '
        'see the original screenshot and return original normalized 0-1 coordinates. '
        'Stop after you have enough evidence. You have at most three turns. '
    )
    if PYTHON_ENABLED:
        system += (
            'You may run arbitrary persistent Python: {"action":"python","code":"print(image.size)"}. '
            'Globals: image (PIL original), instruction, np, Image, ImageDraw. '
            'show_image(pil_image) displays evidence to you. '
            'ground_model(alias,instruction,image=None,frame=None) returns original normalized 0-1 '
            'coordinates if a frame is provided, plus the raw model response. '
            'ImageFrame.full(size) and ImageFrame.crop(original_size, crop_xyxy, supplied_size) '
            'describe coordinate mappings for transformed images. Use print() to read results. '
            'Use only available packages; matplotlib is not installed. '
        )
    messages = [{'role':'system','content':system},{'role':'user','content':[
        {'type':'text','text':f"Find this existing UI target: {ctx['instruction']}\nFirst estimate (0-1): {first['point']}"},
        {'type':'image_url','image_url':{'url':image_url(ctx['path'])}},
    ]}]
    events = []
    with PythonSession(ctx['path'],ctx['instruction'],pool,timeout=45) as session:
        for turn in range(3):
            if pool.calls >= pool.max_calls:
                break
            if turn == 2:
                messages.append({'role':'user','content':'This is the last turn. Return a final point now.'})
                schema=actions[0]
            else:
                schema={'oneOf':actions}
            try:
                raw=pool.chat('qwen',messages,response_schema=schema)
                action=json.loads(raw)
                messages.append({'role':'assistant','content':raw})
                if action['action']=='final':
                    point=to_original_point(action['point'],'norm1000',ImageFrame.full(ctx['image'].size))
                    return {'point':point,'first_point':first['point'],'events':events,
                            'used_python':any(e['type']=='python' for e in events),'raw':raw}
                if action['action']=='model':
                    result=pool.ground(action['model'],ctx['instruction'],image=ctx['path'],
                                       frame=ImageFrame.full(ctx['image'].size))
                    events.append({'type':'model','model':action['model']})
                    feedback=[{'type':'text','text':json.dumps(result)}]
                else:
                    result=session.execute(action['code'])
                    events.append({'type':'python','code':action['code'],'error':result['error'],
                                   'stdout':result['stdout']})
                    feedback=[{'type':'text','text':json.dumps({'stdout':result['stdout'],'error':result['error']})}]
                    feedback.extend({'type':'image_url','image_url':{'url':url}} for url in result['images'])
                messages.append({'role':'user','content':feedback})
            except Exception as error:
                events.append({'type':'failure','error':f'{type(error).__name__}: {error}'})
                break
    return {'point':first['point'],'first_point':first['point'],'events':events,'fallback':True,
            'used_python':any(e['type']=='python' for e in events)}
