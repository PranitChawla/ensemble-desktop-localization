
import pytest

from grounding.coordinates import ImageFrame, normalize_grounding, parse_grounding, to_original_point


@pytest.mark.parametrize("space,point", [
    ("pixels", [960, 270]), ("norm1000", [500, 250]), ("norm01", [0.5, 0.25]),
])
def test_units_map_to_same_original_point(space, point):
    assert to_original_point(point, space, ImageFrame.full((1920, 1080))) == [0.5, 0.25]


def test_resized_region_maps_to_original_not_input_image():
    frame = ImageFrame.crop((2000, 1000), [1000, 100, 1800, 700], (400, 300))
    assert to_original_point([500, 500], "norm1000", frame) == [0.7, 0.4]
    assert to_original_point([200, 150], "pixels", frame) == [0.7, 0.4]


def test_letterbox_offset_is_removed_explicitly():
    # A 2000x1000 screenshot scaled to 1000x500 and padded 250 px above and below.
    frame = ImageFrame((2000, 1000), (1000, 1000), (2, 0, 0, 0, 2, -500))
    assert to_original_point([500, 375], "pixels", frame) == [0.5, 0.25]
    with pytest.raises(ValueError, match="outside"):
        to_original_point([500, 100], "pixels", frame)


@pytest.mark.parametrize("text", [
    '{"point_2d":[750,500]}',
    '[{"label":"Save","point_2d":[750,500]}]',
    '<tool_call>{"arguments":{"coordinate":[750,500]}}</tool_call>',
    '<point>(750,500)</point>',
    '{"bbox_2d":[700,450,800,550]}',
])
def test_supported_native_shapes(text):
    result = normalize_grounding(text, "norm1000", (1920, 1080), ImageFrame.full((1920, 1080)))
    assert result["point_original_norm01"] == [0.75, 0.5]
    assert result["raw_response"] == text


def test_units_are_never_guessed_from_magnitude():
    # Tiny norm1000 values remain tiny: interpreting these as norm01 would silently change the answer.
    result = normalize_grounding('[0.5,0.5]', 'norm1000', (1000, 1000))
    assert result["point_input_norm01"] == [0.0005, 0.0005]
    assert result["point_original_norm01"] is None


def test_multiple_targets_and_out_of_bounds_are_rejected():
    with pytest.raises(ValueError):
        parse_grounding('[{"point":[100,100]},{"point":[900,900]}]')
    with pytest.raises(ValueError):
        parse_grounding('[100,100] and [900,900]')
    with pytest.raises(ValueError):
        normalize_grounding('{"point":[1001,500]}', 'norm1000', (1000, 1000))
    with pytest.raises(ValueError, match="dimensions"):
        normalize_grounding('[500,500]', 'norm1000', (500, 500), ImageFrame.full((1000, 1000)))


def test_refusal_is_explicit():
    result = normalize_grounding('{"point":null}', 'pixels', (1920, 1080))
    assert result["kind"] == "refusal" and result["point_input_norm01"] is None


def test_config_does_not_invent_native_formats():
    from grounding.config import load_config

    cfg = load_config("configs/generalists.json")
    formats = {model["alias"]: model["grounding"] for model in cfg["models"]}
    assert formats["qwen"]["native_space"] == "norm1000"
    for alias in ("ministral", "lfm"):
        assert formats[alias]["native_space"] is None
        assert formats[alias]["status"] == "prompted_not_native"


def test_grounding_requests_use_native_units_and_shared_budget(monkeypatch, tmp_path):
    import httpx
    from PIL import Image

    from grounding.client import BudgetExceeded, ModelPool
    from grounding.config import load_config

    image = tmp_path / 'screenshot.png'
    Image.new('RGB', (1920, 1080)).save(image)

    def post(url, *, json, timeout):
        prompt = json['messages'][0]['content'][0]['text']
        assert '0–1000' in prompt
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'choices': [{'message': {'content': '{"point_2d":[750,500]}'}, 'finish_reason': 'stop'}],
        })
    monkeypatch.setattr(httpx, 'post', post)
    pool = ModelPool(load_config('configs/generalists.json'), max_calls=1)
    result = pool.ground('qwen', 'Save', image=image, frame=ImageFrame.full((1920, 1080)))
    assert result['point_original_norm01'] == [0.75, 0.5]
    assert result['native_space'] == 'norm1000'
    with pytest.raises(BudgetExceeded):
        pool.ask('lfm', 'another request')



def test_jedi_uses_resized_pixel_frame(monkeypatch,tmp_path):
    import httpx
    from PIL import Image

    from grounding.adapters import smart_size
    from grounding.client import ModelPool
    from grounding.config import load_config
    image=tmp_path/'large.png'
    Image.new('RGB',(3840,2160)).save(image)
    width,height=smart_size((3840,2160))
    def post(url,*,json,timeout):
        assert f"{width}x{height}" in json['messages'][0]['content']
        text='{"name":"computer_use","arguments":{"action":"left_click","coordinate":['+str(width/2)+','+str(height/2)+']}}'
        return httpx.Response(200,request=httpx.Request('POST',url),json={
            'choices':[{'message':{'content':'<tool_call>'+text+'</tool_call>'}}]})
    monkeypatch.setattr(httpx,'post',post)
    result=ModelPool(load_config()).ground('jedi','Save',image=image,frame=ImageFrame.full((3840,2160)))
    assert result['processor_image_size']==[width,height]
    assert result['point_original_norm01']==[0.5,0.5]
    assert width%28==height%28==0 and width*height<=2116800


def test_venus_documented_refusal(monkeypatch,tmp_path):
    import httpx
    from PIL import Image

    from grounding.client import ModelPool
    from grounding.config import load_config
    image=tmp_path/'screen.png'
    Image.new('RGB',(400,200)).save(image)
    def post(url,*,json,timeout):
        assert json['messages'][0]['content'][0]['type']=='image_url'
        return httpx.Response(200,request=httpx.Request('POST',url),json={
            'choices':[{'message':{'content':'[-1,-1]'}}]})
    monkeypatch.setattr(httpx,'post',post)
    result=ModelPool(load_config()).ground('venus','absent',image=image,frame=ImageFrame.full((400,200)))
    assert result['point_original_norm01'] is None
