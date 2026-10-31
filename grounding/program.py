"""Run a trusted, agent-authored Python inference program without evaluator labels."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from grounding.coordinates import point_to_input_norm01


def run_program(predict,screenshot,instruction,pool):
    from grounding.study_policies import direct

    path=Path(screenshot).resolve()
    with Image.open(path) as source:
        image=source.convert('RGB')
    with tempfile.TemporaryDirectory(prefix='grounding-program-') as directory:
        ctx={'image':image,'path':path,'instruction':instruction,'pool':pool,'scratch':Path(directory)}
        saved={}
        def baseline(alias):
            if alias not in saved:
                saved[alias]=direct(ctx,alias)
            return saved[alias]
        result=predict(ctx,baseline)
    point=result.get('point')
    if point is not None:
        point=point_to_input_norm01(point,'norm01',image.size)
    return dict(result,status='completed',point=point,calls=pool.calls)
