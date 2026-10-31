# Ensemble GUI Grounding

The optimization agent edits **`submission.py`**, a Python inference program
around three frozen GUI-specialist VLMs. It can manipulate images, sample models,
route requests, aggregate coordinates, and verify candidates. The VLMs return
predictions; they do not generate or execute Python. No particular image operation
or ensemble strategy is imposed.

The current pool is **UI-Venus-1.5-2B (2.44B full parameters), MAI-UI-2B (2.13B),
and Jedi-3B-1080p (3.75B)**. Exact revisions are pinned in `models.json`.
See the [literature sweep](docs/model-sweep.md) and
[coordinate-format audit](docs/model-formats.md). The initial generalist pool is
preserved in `configs/generalists.json`; checkpoint replacement and harness gains
are evaluated separately.

All three servers share one H100 80 GB, with a 0.25 memory fraction each, BF16,
16,384 total positions and one active sequence per server. Calls are serialized.
The memory fractions cover executor weights and context caches; sampled device
memory during real requests determines the practical headroom.

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
Python-initiated helper-model call, and runs the editable Python policy on a synthetic
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

## Editable Python harness

`ground(screenshot, instruction, pool)` in `submission.py` returns a normalized
original-image point or refusal. The optimization agent changes the program between
development evaluations. Python may use PIL, NumPy, imports, temporary files, and
arbitrary computations. The fixed VLMs are accessed through `pool.chat`, `pool.ask`,
and `pool.ground`; all requests share an authoritative budget and failed calls count.

`pool.ground` uses checkpoint-specific native prompts and explicit coordinate
conversions. Transformed images require an `ImageFrame`. Units are never inferred
from numerical magnitude. The trusted research evaluator does not provide
adversarial isolation from dataset files; a hardened task judge needs that boundary.

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
This is an initial working harness, not yet a packaged OpenRSI/Harbor task or
an autonomous outer hill-climbing runner. No training is performed. The bounded empirical study is in progress; final measured findings will be recorded here.

