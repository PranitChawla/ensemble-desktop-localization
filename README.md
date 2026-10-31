# Ensemble GUI Grounding

An initial research environment for open-ended GUI grounding with a persistent
Python tool and a diverse, frozen VLM pool. The editable inference policy lives
in **`submission.py`**. It can write arbitrary Python, call any of the available
models, display intermediate images, and choose its own strategy and stopping rule.
No crop pipeline or fixed ensemble is imposed.
See [`docs/model-formats.md`](docs/model-formats.md) for the documentation/code audit
of each checkpoint's native coordinate convention and the explicit fallback protocols.
The [initial H100 validation](docs/validation.md) records measured memory, context
lengths, Python-tool checks, and the small visible integration result.

## Model pool and memory

The initial pool uses three independently pretrained language backbones:

| Alias | Checkpoint | Full parameters | Approximate BF16 weights |
| --- | --- | ---: | ---: |
| `qwen` | `Qwen/Qwen3.5-2B` | 2.27B | 4.5 GB |
| `ministral` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 3.85B | 7.7 GB |
| `lfm` | `LiquidAI/LFM2-VL-3B` | 3.00B | 6.0 GB |

`models.json` pins exact checkpoint commits. All three servers run on **one H100
80 GB**, with a 0.25 GPU-memory fraction per server (75% combined), BF16,
16,384 total model positions, one active sequence per server, bounded visual
inputs, and eager execution. Calls are serialized across the pool. The memory
fraction includes each model executor's weights and automatically sized context
cache; it is not a hard process-level isolation boundary.

## Local development

```bash
uv venv --python 3.12
uv pip install -e '.[cloud,dev]'
source .venv/bin/activate
pytest -q
gui-grounding server-commands
```

The local package does not install CUDA, PyTorch, or vLLM. The pinned Modal image
provides these. Local evaluation assumes compatible model servers already exist.

## Modal H100 smoke test

Run with your authenticated Modal CLI configuration.
The Modal app and H100 function are both named **`ensemble-desktop-localization`**;
the GitHub repository remains **`ensemble-desktop-localization`**.

```bash
modal run modal_app.py --download-only
modal run modal_app.py --skip-download
```

The first command caches pinned checkpoints on a CPU container. The second
starts one H100 container, launches servers sequentially, runs screenshot and
long-context requests through each model, exercises persistent Python and a
Python-initiated helper-model call, and runs the editable agent on a synthetic
desktop screenshot. It records actual prompt tokens, GPU-memory snapshots,
model responses, tool steps, and server logs.
GPU memory is also sampled every second throughout startup and inference; the report
includes the sampled peak and minimum free memory, rather than only idle snapshots.

These are **plumbing tests, not ScreenSpot-Pro benchmark results**. The long-context
probe reports its actual prompt length; it does not prove arbitrary 16K multimodal
conversations fit. Stress the intended maximum image sizes, history, and outputs
before increasing concurrency or adding a fourth server.

There is no public HTTP endpoint or always-on deployment. All model servers bind
to container loopback. The H100 function has a 50-minute maximum duration and
the CLI run stops after the experiment. A completed test does not leave an idle
GPU serving indefinitely.

Persistent volumes created by the application:

- `ensemble-desktop-localization-hf-v1`: checkpoint download cache.
- `ensemble-desktop-localization-results-v1`: reports, traces, and server logs by run ID.

The CLI also saves the report under local `outputs/` (ignored by Git). Fetch full
remote logs with:

```bash
modal volume get ensemble-desktop-localization-results-v1 /<run-id> ./results/<run-id>
```

## Python tool contract

The controller emits a JSON `python` action or a `final` action. Python runs in
a fresh persistent interpreter per example with PIL, NumPy, the screenshot,
instruction, and two helpers:

```python
answer = call_model('ministral', 'Describe the relevant control.', image=image)
location = ground_model('qwen', instruction)
print(location['point_original_norm01'])
show_image(image)
```

Imports, algorithms, transformations, annotations, and intermediate files are
open-ended. The parent accounts for controller and helper calls under one budget;
`ground_model` preserves each configured coordinate convention and returns explicit
input/original-frame coordinates plus the raw response. For transformed images,
provide an `ImageFrame`; coordinate units are never guessed from numerical magnitude.
failed requests count and are not silently retried. Python execution has a timeout,
output bounds, a disposable working directory, no inherited account credentials,
and no CUDA devices exposed by default.

**This interpreter is not a security sandbox for adversarial code.** It shares the
research container's network and filesystem. Use only trusted research candidates
in disposable containers without mounted secrets. A hardened benchmark Judge
requires an additional isolation and request-accounting boundary.

## Visible development evaluation

Create a JSONL manifest using paths relative to the manifest file:

```json
{"id":"example-1","image":"images/screen.png","instruction":"Click Save","bbox_xyxy":[100,200,180,240]}
```

`bbox_xyxy` uses **original-image pixel coordinates**. Predictions use normalized
`[x,y]` coordinates. A `null` box denotes a refusal example. Labels remain in the
evaluator and are not passed to the agent. Operational failures make aggregate
accuracy unavailable, while traces retain the specific failure.

```bash
gui-grounding evaluate data/visible.jsonl --output results/experiment-001 --limit 20
```

ScreenSpot-Pro is the intended visible development benchmark. Split development
and validation by screenshot, and consider application-disjoint validation.
**OSWorld-G is reserved for later transfer evaluation and is not downloaded or
queried by these scripts.** It is a public held-out benchmark, not a secret dataset.
Pin its original/refined instruction protocol before introducing a final evaluator.

## Research boundary

Change `submission.py` to test hypotheses about Python use, prompts, model routing,
verification, representations, memory, and stopping. For matched comparisons, keep
the model commits, serving settings, data splits, evaluator, and budgets fixed.
This is an initial working environment, not yet a packaged OpenRSI/Harbor task or
an autonomous outer hill-climbing runner. No training or benchmark gain is claimed.

