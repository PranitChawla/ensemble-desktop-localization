# Build a reliable GUI-grounding harness

You are the coding and research agent. Improve `submission.py`, a Python inference
program that locates an existing UI element from a screenshot and a natural-language
instruction. You have a Python tool for development and three frozen VLM services.
You write the algorithm; the VLMs return evidence and predictions. Their responses
are never executed as Python.

Your objective is to maximize point-in-box accuracy under the fixed task budget.
Use the visible ScreenSpot-Pro set to develop the program. OSWorld-G is held out
for the judge. Keep the model checkpoints, serving configuration, evaluator, task
contract, and dependencies fixed during an improvement run.

## Deliverable

Submit one self-contained file, `submission.py`, exporting:

```python
def ground(screenshot: str, instruction: str, pool) -> dict:
    # screenshot is a local PNG of the original screen.
    return {"point": [0.5, 0.5]}
```

Return a finite `[x,y]` point in **original-image [0,1] coordinates**, or `None` when
the requested target is absent. Optional JSON-serializable diagnostic fields are
allowed within a 64 KB JSON result. Captured stdout is limited to 16 KB. Do not return a bounding box or coordinates in an intermediate image.
The screenshot is static: no clicking, application execution, or browser interaction
is required. Variables and temporary images may persist within one example; each
example starts a fresh submission process.

## VLM API

`pool.models` describes the fixed aliases, roles, revisions, and grounding formats.
The default pool contains `venus`, `mai`, and the general VLM `qwen`.

- `pool.ask_vlm(alias, question, image=path_or_PIL, ...) -> str`: ask **any** visual
  question. Classification, visible text reading, semantic interpretation,
  verification, instruction decomposition, and grounding are all permitted.
  `images=[...]` accepts several images created or selected by your Python program.
  `response_schema` optionally constrains JSON; `temperature`, `top_p`, and
  `max_tokens` control generation within the fixed limits.
- `pool.chat(alias, messages, ...) -> str`: lower-level OpenAI-style messages for
  custom prompts and conversations. Use local image data, not remote image URLs.
- `pool.ground(alias, instruction, image=..., frame=...) -> dict`: convenience
  adapter using that checkpoint's documented prompt and coordinate convention.
  Returns raw text, input-relative coordinates, and original-relative coordinates
  when the image frame is supplied. This method does not restrict the general API.
- `pool.calls` and `pool.max_calls` report the shared request budget. The evaluator
  owns the actual counter; changing submission-side attributes cannot reset it.

You may build any Python decision tree around these calls. No particular image
operation, ensemble, sampling scheme, routing taxonomy, or verification rule is
required. You may manipulate images, compute features, generate visual evidence,
ask conditional questions, and choose when to stop. Use the available Python
standard library, Pillow, NumPy, and fixed harness utilities. Do not train models,
load additional checkpoints, install inference dependencies, or call outside AI
services.

## Coordinates

Read `docs/model-formats.md`. Units are not inferred from the magnitude of numbers.
Use `ImageFrame.full(original_size)` for an original image. For a resized region,
use `ImageFrame.crop(original_size, region_xyxy, supplied_image_size)`. General
transforms require an explicit mapping, and nonlinear transforms need your own
inverse mapping. The final point must always refer to the original screenshot.

## Fixed budget

The authoritative values live in `task.toml` and `models.json`:

- Six model requests total per example, across all models and question types.
- At most 512 generated tokens per request, 45 seconds per request, and 120 seconds
  for submission execution, including model calls.
- At most three images per request. Per-model image limits are fixed in
  `max_request_pixels`; only a bounded `max_pixels` override is permitted.
- The servers share one H100, with serialized requests and fixed context capacity.
- Failed HTTP requests consume budget. There are no hidden retries. A handled
  failure may fall back to an existing prediction; uncaught errors, timeouts, and
  invalid outputs count as misses.

## Improvement loop

1. State a concrete hypothesis and edit `submission.py`.
2. Evaluate matched visible examples in a fresh output directory.
3. Inspect errors, accuracy, request counts, tokens, and timings. You may inspect
   visible development labels and feedback to understand failures.
4. Keep or revert the program based on evidence. Preserve its source hash and logs.
5. Submit the final file. Do not inspect or tune against held-out examples.

The inference program must use only the screenshot, instruction, public model
metadata, and responses to its own budgeted queries. Do not read annotations,
benchmark identifiers, evaluator memory, prior example answers, or external answer
caches during inference. Do not bypass the provided VLM client or alter task files.
The judge owns labels and scoring; the submission does not select its own score.
