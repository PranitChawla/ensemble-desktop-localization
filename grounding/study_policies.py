"""Controlled research policies; all receive image/instruction, never target labels."""
from __future__ import annotations

import json
import re

from PIL import ImageDraw, ImageFont

from grounding.client import image_url
from grounding.coordinates import ImageFrame, grounding_prompt, normalize_grounding
from grounding.tools import PythonSession
from submission import parse_action


def direct(ctx, alias="qwen", space=None, bbox=False, request_pixels=None):
    image = ctx["image"]
    pool = ctx["pool"]
    space = space or pool.models[alias]["grounding"]["request_space"]
    prompt = grounding_prompt(ctx["instruction"], space, image.size)
    if bbox:
        prompt = prompt.replace('its center', 'its bounding box').replace(
            '{"point": [x,y]}', '{"bbox_2d": [x_min,y_min,x_max,y_max]}')
    raw = pool.ask(alias, prompt, image=ctx["path"],
                   processor_kwargs={'max_pixels':request_pixels} if request_pixels else None)
    result = normalize_grounding(raw, space, image.size, ImageFrame.full(image.size))
    return {"point": result["point_original_norm01"], "raw": raw, "space": space}


def repeat_refine(ctx, first, *, alias="qwen", mode="full", fraction=0.3,
                  fallback=False, max_displacement=None):
    image, pool = ctx["image"], ctx["pool"]
    point = first["point"]
    if point is None:
        return first
    width, height = image.size
    if mode == "full":
        prompt = (grounding_prompt(ctx["instruction"], "norm1000", image.size) +
                  f"\nAn initial estimate is [{point[0]*1000:.1f},{point[1]*1000:.1f}]. "
                  "Check the screenshot carefully and correct the estimate if needed.")
        raw = pool.ask(alias, prompt, image=ctx["path"])
        result = normalize_grounding(raw, "norm1000", image.size, ImageFrame.full(image.size))
        return {"point": result["point_original_norm01"], "raw": raw}
    # Python selects, renders, and maps a higher-detail view; no target label enters this computation.
    x, y = point[0] * width, point[1] * height
    region_width, region_height = max(256, width * fraction), max(256, height * fraction)
    x0 = max(0, min(width - region_width, x - region_width / 2))
    y0 = max(0, min(height - region_height, y - region_height / 2))
    box = tuple(map(int, (x0, y0, min(width, x0 + region_width), min(height, y0 + region_height))))
    region = image.crop(box)
    path = ctx["scratch"] / "detail.png"
    region.save(path)
    frame = ImageFrame.crop(image.size, box, region.size)
    prompt = grounding_prompt(ctx["instruction"], "norm1000", region.size)
    if mode=='detail-context':
        messages=[{'role':'user','content':[
            {'type':'text','text':'The first image gives full-screen context. The second image is a detail view. '+
             prompt+' Return coordinates in the SECOND image only.'},
            {'type':'image_url','image_url':{'url':image_url(ctx['path'])}},
            {'type':'image_url','image_url':{'url':image_url(path)}}]}]
        raw=pool.chat(alias,messages)
    else:
        raw = pool.ask(alias, prompt, image=path)
    fallback_reason=None
    try:
        result = normalize_grounding(raw, "norm1000", region.size, frame)
        final=result['point_original_norm01']
    except ValueError as error:
        if not fallback:
            raise
        final, fallback_reason = point, str(error)
    if fallback and final is None:
        final, fallback_reason = point, 'refusal_in_detail'
    if final is not None and max_displacement is not None:
        displacement=((final[0]-point[0])**2+(final[1]-point[1])**2)**0.5
        if displacement>max_displacement:
            final,fallback_reason=point,'large_displacement'
    return {"point": final, "raw": raw, "region": list(box), "fallback_reason":fallback_reason,
            "python_operations": ["select_region", "render", "coordinate_mapping"]}


def select_candidate(ctx, candidates, *, judge="qwen", annotated=True):
    valid = [(alias, value["point"]) for alias, value in candidates.items() if value.get("point") is not None]
    if not valid:
        return {"point": None}
    image = ctx["image"].copy()
    width, height = image.size
    descriptions = []
    if annotated:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(24, width // 80))
        except OSError:
            font = ImageFont.load_default(size=24)
    colors = ['#ff0000', '#00aa00', '#ff00ff']
    for number, (alias, point) in enumerate(valid, 1):
        x, y = int(point[0] * width), int(point[1] * height)
        descriptions.append(f"{number}: ({x}, {y}) pixels")
        if annotated:
            radius = max(10, width // 150)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=colors[(number-1) % 3], width=4)
            draw.text((x+radius+3, y-radius-8), str(number), fill=colors[(number-1) % 3], font=font,
                      stroke_width=2, stroke_fill="white")
    path = ctx["scratch"] / "candidates.png"
    if annotated:
        image.save(path)
    prompt = (f"Instruction: {ctx['instruction']}\nWhich candidate location targets the requested UI control?\n" +
              '\n'.join(descriptions) + '\nReturn only JSON {"choice": 1} with the best candidate number. ' +
              ('Numbered circles mark the candidate positions in the image.' if annotated else 'Inspect the original image.'))
    raw = ctx["pool"].ask(judge, prompt, image=path if annotated else ctx["path"])
    match = re.search(r'"choice"\s*:\s*(\d+)', raw)
    if not match:
        raise ValueError('Candidate selector did not return a choice')
    choice = int(match[1])
    if not 1 <= choice <= len(valid):
        raise ValueError('Candidate selector returned an invalid choice')
    return {"point": valid[choice-1][1], "selected_alias": valid[choice-1][0], "raw": raw,
            "python_operations": ["annotate_candidates"] if annotated else []}


def agent(ctx, *, controller="qwen", python=True, force_python=False, max_steps=5, structured=False):
    pool = ctx['pool']
    instruction = ctx['instruction']
    system = (
        "Locate an EXISTING UI target in the supplied static screenshot. Return its position, not an application. "
        'Return one JSON action each turn. Final: {"action":"final","point":[x,y],"coordinate_space":"norm1000"}. '
        'Use null for an absent target. You may query a helper with {"action":"model","model":"qwen"}. '
        'Available helpers are qwen, ministral, lfm; a helper sees the original screenshot and instruction. '
        'All calls share the same budget. Use evidence and stop when sufficiently certain. '
    )
    if python:
        system += (
            'You also have arbitrary persistent Python: {"action":"python","code":"..."}. '
            'Globals: image (PIL screenshot), instruction, np, Image, ImageDraw, models, grounding_formats. '
            'Helpers: ground_model(alias, instruction, image=None, frame=None), '
            'call_model(alias,prompt,image=None), show_image(image), '
            'ImageFrame.full(size), ImageFrame.crop(original_size, crop_xyxy, supplied_size). '
            'ground_model returns point_original_norm01 and raw_response. A transformed image needs its ImageFrame. '
        )
        if force_python:
            system += 'Before finalizing, execute at least one useful Python program to inspect or verify the target. '
    messages = [{'role':'system','content':system}, {'role':'user','content':[
        {'type':'text','text':instruction}, {'type':'image_url','image_url':{'url':image_url(ctx['path'])}}]}]
    events, used_python = [], False
    schema={'type':'object','properties':{
        'action':{'type':'string','enum':['final','model','python'] if python else ['final','model']},
        'point':{'anyOf':[{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},{'type':'null'}]},
        'coordinate_space':{'type':'string','enum':['norm1000','norm01','pixels']},
        'model':{'type':'string','enum':['qwen','ministral','lfm']},'code':{'type':'string'}},
        'required':['action'],'additionalProperties':False}
    with PythonSession(ctx['path'], instruction, pool, timeout=75) as session:
        for _ in range(max_steps):
            raw = pool.chat(controller, messages, response_schema=schema if structured else None)
            messages.append({'role':'assistant','content':raw})
            try:
                try:
                    action = parse_action(raw)
                except ValueError:
                    match = re.search(r'\{[^{}]*"action"\s*:\s*"model"[^{}]*\}', raw)
                    if not match:
                        raise
                    action = json.loads(match[0])
                if action['action'] == 'final':
                    if force_python and not used_python:
                        messages.append({'role':'user','content':'Execute a useful Python check before finalizing.'})
                        continue
                    from grounding.coordinates import to_original_point
                    point = action.get('point')
                    if point is not None:
                        point = to_original_point(point, action.get('coordinate_space','norm1000'),
                                                  ImageFrame.full(ctx['image'].size))
                    return {'point':point,'events':events,'used_python':used_python,'raw':raw}
                if action['action'] == 'model':
                    response = pool.ground(action['model'], instruction, image=ctx['path'],
                                           frame=ImageFrame.full(ctx['image'].size))
                    events.append({'type':'model','model':action['model'],'result':response})
                    messages.append({'role':'user','content':json.dumps(response)})
                elif action['action'] == 'python' and python:
                    used_python = True
                    response = session.execute(action['code'])
                    events.append({'type':'python','code':action['code'],'stdout':response['stdout'],
                                   'error':response['error']})
                    parts = [{'type':'text','text':json.dumps({'stdout':response['stdout'][:3000],
                                                              'error':response['error']})}]
                    parts.extend({'type':'image_url','image_url':{'url':url}} for url in response['images'])
                    for prior in messages[2:]:
                        if isinstance(prior.get('content'),list):
                            prior['content'] = [p for p in prior['content'] if p['type'] != 'image_url']
                    messages.append({'role':'user','content':parts})
                else:
                    raise ValueError('Action is not available')
            except (ValueError, KeyError) as error:
                events.append({'type':'invalid_action','error':str(error)})
                messages.append({'role':'user','content':f'Invalid action: {error}. Correct the JSON.'})
    return {'point':None,'events':events,'used_python':used_python,'status':'step_limit'}

