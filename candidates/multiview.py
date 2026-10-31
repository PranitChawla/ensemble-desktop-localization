"""Five-call Python program: inspect four regions, then ground in a detail montage."""

from grounding import study_policies as policies


def predict(ctx, baseline):
    from PIL import Image

    from grounding.coordinates import ImageFrame

    original=ctx['image']
    width,height=original.size
    proposals=[]
    for row in range(2):
        for column in range(2):
            box=(column*width//2,row*height//2,(column+1)*width//2,(row+1)*height//2)
            tile=original.crop(box)
            path=ctx['scratch']/f'tile-{row}-{column}.png'
            tile.save(path)
            local=dict(ctx,image=tile,path=path)
            try:
                result=policies.direct(local,'qwen')
                point=result['point']
                if point is not None:
                    proposals.append([(box[0]+point[0]*tile.width)/width,
                                      (box[1]+point[1]*tile.height)/height])
            except ValueError:
                pass
    if not proposals:
        return {'point':None,'python_operations':['four_view_search']}
    panel_size=512
    collage=Image.new('RGB',(1024,1024),'#eeeeee')
    boxes=[]
    detail_size=max(256,min(width,height)*0.25)
    for index,point in enumerate(proposals):
        x,y=point[0]*width,point[1]*height
        x0=max(0,min(width-detail_size,x-detail_size/2))
        y0=max(0,min(height-detail_size,y-detail_size/2))
        box=tuple(map(int,(x0,y0,min(width,x0+detail_size),min(height,y0+detail_size))))
        boxes.append(box)
        patch=original.crop(box).resize((panel_size,panel_size))
        collage.paste(patch,((index%2)*panel_size,(index//2)*panel_size))
    path=ctx['scratch']/'montage.png'
    collage.save(path)
    local=dict(ctx,image=collage,path=path,
               instruction=ctx['instruction']+' (The image contains independent detail panels from one screen.)')
    selected=policies.direct(local,'qwen')
    if selected['point'] is None:
        return {'point':None,'proposals':proposals,'python_operations':['four_view_search','montage']}
    cx,cy=selected['point'][0]*1024,selected['point'][1]*1024
    column,row=min(1,int(cx//512)),min(1,int(cy//512))
    index=row*2+column
    if index>=len(boxes):
        return {'point':None,'proposals':proposals,'invalid_empty_panel':True}
    point=[(cx-column*512)/512,(cy-row*512)/512]
    frame=ImageFrame.crop(original.size,boxes[index],(512,512))
    from grounding.coordinates import to_original_point
    return {'point':to_original_point(point,'norm01',frame),'proposals':proposals,'detail_boxes':boxes,
            'raw':selected['raw'],'python_operations':['four_view_search','montage','coordinate_mapping']}
