# Submission API

The evaluator calls `ground(screenshot, instruction, pool)` in a fresh Python
process. `screenshot` is the original RGB PNG, with a generic filename. The process
receives no example ID, application label, target type, or ground-truth box.

## General questions

```python
text = pool.ask_vlm(
    "qwen", "Read the visible text on the selected dialog.",
    image=pil_image, max_tokens=128, temperature=0,
)
```

The answer is a string. The API does not assume that the question is a grounding
task and does not parse arbitrary answers as coordinates. `image` accepts a local
path or PIL image; `images` accepts a list instead. A Python program may construct
its own visual representations before asking a question. The two image arguments
are mutually exclusive.

Optional arguments are `response_schema`, `max_tokens`, `temperature`, `top_p`, and
`processor_kwargs`. JSON schemas constrain generation but do not establish that an
answer is correct. Decode and check answers before using them in a decision tree.
Only `max_pixels` may be overridden, up to the model's fixed `max_request_pixels`.

`pool.chat(alias, messages, ...)` accepts custom OpenAI-style conversation messages.
For image content, use local data URLs; `grounding.client.encode_image(pil_image)`
and `image_url(path)` construct them. Remote URLs and extra image types are outside
the task interface. Calls are serialized across the three servers.

## Native grounding convenience

```python
from grounding.coordinates import ImageFrame

answer = pool.ground(
    "venus", instruction, image=pil_image,
    frame=ImageFrame.full(pil_image.size),
    processor_kwargs={"max_pixels": 4194304},
)
point = answer["point_original_norm01"]
```

The adapter preserves `raw_response`, `coordinate_space`, `native_space`,
`point_input_norm01`, and `point_original_norm01`. The last field requires a known
frame. A `None` point may represent a documented refusal; malformed or ambiguous
outputs raise an error. See `model-formats.md` for checkpoint-specific caveats.

For a resized region:

```python
region = image.crop((x0, y0, x1, y1)).resize((800, 600))
frame = ImageFrame.crop(image.size, (x0, y0, x1, y1), region.size)
answer = pool.ground("venus", instruction, image=region, frame=frame)
```

This is an example mapping, not a required algorithm. General affine transforms
can construct `ImageFrame` directly. A nonlinear transform needs a caller-owned
inverse map before returning a point in the original image.

## Budgets and failures

`pool.calls` and `pool.max_calls` expose the current shared request count. The parent
evaluator owns enforcement, including requests initiated from multiple submission
threads. Attempted HTTP requests consume budget even when they fail. Invalid local
arguments do not send a request. There are no automatic retries; explicit retries
consume another request. Generation requests are capped by the task output limit.

Model errors reach the submission as exceptions. A program may catch them and use
an existing prediction. Uncaught exceptions, execution timeouts, non-JSON results,
and invalid points become misses. Scores and costs are computed by the evaluator,
not by fields returned from the submission.

Output must be a JSON-serializable dictionary with `point`, using finite numbers
in [0,1] or `None`. The JSON result is limited to 64 KB; captured stdout/stderr is
limited to 16 KB. A fresh interpreter, temporary directory, and module are created
for each example, and all subprocesses in its process group are stopped afterwards.
