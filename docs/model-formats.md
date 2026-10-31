# Grounding format audit

Reviewed 2026-10-31 for the **exact pinned checkpoints in `models.json`**.
Coordinate conventions are model-specific. An image processor's resize policy or
an unrelated OCR product's bounding-box schema does not establish a VLM's native
coordinate convention.

| Model | Documented native convention | Harness grounding request |
| --- | --- | --- |
| Qwen3.5-2B | Relative **0–1000**, x then y; xyxy for boxes | 0–1000 point, preserving native coordinate units |
| Ministral 3 3B Instruct BF16 | **Not specified in the reviewed sources** | Explicitly requested original supplied-image pixels; experimental prompted protocol |
| LFM2-VL-3B | **Not specified in the reviewed sources** | Explicitly requested 0–1 relative point; experimental prompted protocol |

## Qwen

The Qwen team's [accepted answer about Qwen3.5 coordinates](https://github.com/QwenLM/Qwen3.8/discussions/56)
specifies 0–1000 relative coordinates and links its official
[2D grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb).
The repository has been renamed since those discussions were created; the question
and answer explicitly concern Qwen3.5. We read the notebook code: its point and box
visualizers multiply x by input-image width / 1000 and y by height / 1000.
It uses `point_2d` for points and `bbox_2d` for xyxy boxes. The notebook also explicitly
distinguishes Qwen2.5-VL's absolute coordinates from Qwen3-VL's relative coordinates.

Also read the exact [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md),
configuration, and chat template. The model card alone does not specify the units;
the official Qwen explanation and executable visualization code supply that evidence.

## Ministral

Read the pinned checkpoint's
[model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/blob/b6d637bef2393152b3da2b2fde72eecdee30557e/README.md),
[configuration](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/blob/b6d637bef2393152b3da2b2fde72eecdee30557e/config.json),
and [chat-template code](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/blob/b6d637bef2393152b3da2b2fde72eecdee30557e/chat_template.jinja),
as well as the publisher's [FP8 sibling card and inference examples](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).
These describe visual question answering, JSON, and function calling, but do not
define a dedicated GUI-grounding schema or pixel/relative coordinate convention.

Consequently `native_space` is **null**. The adapter asks for pixels, states the
actual supplied image's width and height, validates bounds, and preserves the raw
response. This is an explicit harness protocol, **not a claim about native training**.
Calibration/accuracy of this prompted protocol remains an empirical research question.

## LFM

Read the exact
[model card](https://huggingface.co/LiquidAI/LFM2-VL-3B/blob/97a81f5c18b73352134099952a4856e2553d42d7/README.md),
[configuration](https://huggingface.co/LiquidAI/LFM2-VL-3B/blob/97a81f5c18b73352134099952a4856e2553d42d7/config.json),
and [chat-template code](https://huggingface.co/LiquidAI/LFM2-VL-3B/blob/97a81f5c18b73352134099952a4856e2553d42d7/chat_template.jinja).
Also inspected the upstream Transformers
[`Lfm2VlProcessor`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_vl/processing_lfm2_vl.py):
it prepares text/image tokens; the inspected processor does not supply a grounding
postprocessor or a coordinate convention. The model card documents tiling and image
tokens, which does not by itself determine output units.

`native_space` is therefore **null**, and the adapter explicitly requests a 0–1
relative point. We do not transfer format claims from a different Liquid checkpoint
or the later LFM2.5-VL family onto this pinned LFM2-VL checkpoint.

## Conversion and framing

`grounding/coordinates.py` supports declared `pixels`, `norm1000`, and `norm01`
units, JSON points and xyxy boxes, supported tool-call wrappers, and point/box tags.
The center of a returned box is the click candidate. It does not infer units from
the size of the numbers, silently clip out-of-bounds points, or select between
multiple returned targets.

`ImageFrame` explicitly maps the supplied image to the original screenshot.
Full screenshots use identity mapping; resized regions use scale and translation;
letterboxing/rotation can use a supplied affine matrix. A transformed image with
no frame yields only an input-image-relative point, never a falsely labeled
original-screenshot point. Arbitrary nonlinear transforms require caller-owned
mapping; an affine matrix cannot represent every possible Python image operation.

The Python tool exposes `ground_model`, `grounding_formats`, `ImageFrame`, and
`to_original_point`. Free-form `call_model` is still available for general reasoning;
its text is not silently interpreted as coordinates.

