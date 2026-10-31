# Three-H100-hour feasibility study

Protocol committed before the GPU experiments. The task remains unrestricted Python-
tool-assisted GUI grounding; individual policies are experimental treatments.

## Budget and data

- At most two independent single-H100 workers, with 5,100 seconds of active function
  time each and 5,300-second hard function timeouts. The remaining allowance covers
  container bootstrap and cleanup; aggregate allocation stays below three H100-hours.
- Startup, idle orchestration, and inference all count toward the GPU budget. CPU-only
  dataset preparation does not. No model training or additional model downloads.
- Pinned ScreenSpot-Pro revision and model revisions from this repository.
- Seed 20261031. Within each application, screenshots are partitioned before sampling.
  Select 128 development examples and 512 reserved validation examples by round robin
  across application/UI-type strata. Exclude the prior smoke-test screenshots from
  validation. Record manifest hashes and all sample identifiers.
- The first 256 validation examples form the primary confirmatory set. The next 256
  may extend precision if time remains, with the policy still frozen. No OSWorld-G.

## Questions and controls

1. Which models localize accurately, and on which applications, target sizes, and
   text/icon categories? Compare direct queries. Probe alternative explicitly prompted
   units for models without a documented native convention, using development only.
2. Do other models recover Qwen's mistakes? Report oracle union as a diagnostic upper
   bound, never as an attainable policy score or a source of inference-time answers.
3. Does making arbitrary Python available help small controllers? Compare the same
   controller with/without Python and a required-Python treatment. Both retain access
   to the same helper models, maximum six model calls, maximum five action turns,
   generation limit, and serving configuration. Report actual use, errors, cost, and
   selection effects; availability and execution are different interventions.
4. Can concrete Python computations help where unrestricted agents fail? Compare
   image annotation/candidate selection against text-only candidate selection. Compare
   a Python-rendered higher-detail view against another full-image refinement query.
   These controls distinguish useful computation from simply spending more calls.

Each evaluated policy operates on a single H100 with three resident servers. The two
workers accelerate independent experiments; they do not give any policy two GPUs.

## Evaluation and selection

- Point-in-original-box accuracy, with fixed examples and no coordinate clipping.
- Invalid model coordinates, exhausted budgets, failed programs, and unresolved action
  limits count as unsuccessful examples. Operational errors are separately counted;
  do not silently remove them from denominators or resample outputs.
- Log prompt/output tokens, actual model requests, request times, Python operations,
  controller/program failures, and sampled GPU-memory peaks.
- Freeze selected policies using development evidence only. Evaluate them and their
  controls on the reserved validation set once selection is frozen. Do not use
  validation examples to design or revise a policy.
- Reuse deterministic first-pass predictions across policy ablations to save compute.
  Count their original calls/tokens/time in logical policy cost; separately report
  actual newly executed requests. Timing with replayed first passes is an estimate of
  sequential model work, not a measured uncached end-to-end latency benchmark.
- Report paired improvements and screenshot-cluster bootstrap intervals, together
  with sample sizes and disagreements. Subgroup findings are descriptive with small
  counts; this is not a full benchmark or a claim about OSWorld-G transfer.
- Preserve all trials, including negative results. Any incomplete budget-limited job
  must be labeled incomplete and compared only on an explicit matched prefix.

The initial broad ablations are specified in `research_app.initial_jobs`. Development
results may motivate additional trials inside the same resource and data boundary.

## Development pilot adjustment

The initial one-megapixel Qwen baseline scored 16/128, with only 3/85 tiny targets
correct. Before opening validation, add a four-megapixel input control, a bounding-box
prompt, and a detail-plus-context treatment. This tests whether visual detail limits
the starting baseline. Processor overrides are logged in request traces.

The initial unconstrained action-JSON controller is also expensive. Further paired
agent trials use the same constrained JSON action grammar, four-call cap, three-turn
cap, and 256-output-token cap in both Python/no-Python arms. The initial unconstrained
pilot is retained as a separate diagnostic, not pooled into those comparisons.
Remaining planned pilot jobs are deferred into the adjusted run. Both runs' measured
GPU time counts against the same three-hour total; no validation feedback motivates
this adjustment.


## Sampling extension requested during development

Before any validation evaluation, add five stochastic proposals (temperature 0.7,
top-p 0.9), followed by one deterministic Qwen selection query. Compare five Qwen
samples against three Qwen + one Ministral + one LFM sample with the same six-call
maximum. Also score a Python coordinate consensus rule from the same proposals:
choose an observed point with the most neighbors within normalized distance 0.03,
breaking ties by total distance. Report proposal oracle coverage separately from
attainable consensus/selection accuracy. Proposal failures remain in the denominator;
there is no retry to fill a failed sample. Store individual proposals and sampling
parameters. This is a finite-sampling experiment, not a best-of-N result using labels.

To prioritize these tests, the remaining Ministral action-controller pilots use 16
examples per arm; the Qwen matched pilots retain 32. The required-Python Ministral
pilot is deferred. Candidate-selection and four-view pilot tests use 32 examples.
These development sample sizes are exploratory. No validation data informed this change.

## Scope correction and specialist-pool phase

The user's clarified target is an outer optimization agent editing the Python
inference harness. VLM-driven Python execution was a mistaken exploratory branch;
its completed pilots remain archived and are excluded from the main question.
All pending controller jobs were removed. The current `submission.py` executes an
agent-authored program without asking a VLM to generate Python.

After development-only generalist failures and the primary-source literature sweep,
replace the pool with UI-Venus-1.5-2B, MAI-UI-2B, and Jedi-3B-1080p. All remain frozen.
This is a new pool phase; its harness gains are measured against its own single-call
baselines. The total GPU budget includes the earlier pilots and cold starts.

Matched sampling now compares five Venus proposals, five MAI proposals, and the
fixed mixed schedule Venus/Jedi/Venus/Jedi/MAI, followed by the same deterministic
MAI candidate selector. Temperature 0.7/top-p 0.9 and consensus rule remain fixed.
Per-request processor overrides are explicitly enabled on trusted loopback servers;
the rejected earlier four-megapixel requests are serving failures, not accuracy
measurements. No reserved validation examples have been evaluated at this point.

The first specialist direct passes favor MAI over Jedi (49/128 versus 28/128).
Before the mixed sampling job starts and before validation, allocate its five
proposals as Venus/MAI/Venus/MAI/Jedi, retaining the same deterministic MAI sixth
selection call. This is a development-informed harness allocation change, not an
increase in the call budget. The simpler three-model coordinate-agreement policy
can be explored offline from development predictions, with parameters frozen before
validation. Remaining annotated/text selector development pilots use 32 examples.

## Stronger resolution baseline and matched serving checks

The trusted request flag alone was insufficient: vLLM rejected large images that
exceeded its preallocated encoder cache. On worker 0, restart only Venus with a
four-megapixel encoder capacity and 4096 batched tokens. Keep default requests at
2116800 pixels explicitly; four-megapixel requests specify 4194304. Native two-
megapixel accuracy remains 51/128 after rerunning all development cases, although
some numerical point outputs differ. Refresh its first-pass cache before further
refinement comparisons. Match worker 1's Venus capacity before validation. The
actual transition code, settings, and memory snapshots are retained in job artifacts.

The stronger one-call Venus baseline scores 71/128 at four megapixels. Before
validation, add matched two-call full-image and Python detail-view policies that
both begin with a fresh four-megapixel native query. Keep the detail fraction at
0.3 and inspect displacement-gating parameters on development only. Test sampling
at this stronger operating point as well: the Venus proposals use four megapixels;
MAI/Jedi and the MAI selector retain their fixed native two-megapixel limits.
Eight-case development checks verify the new sampling request path; they are
plumbing checks rather than selection evidence. MAI-only sampling and the remaining
context-view pilots are deferred to preserve confirmatory compute.

The four-megapixel raw detail policy scored 58/128, below its own first pass
(71/128). Displacement cutoffs 0, .001, .002, .003, .005, .01, .02, .03, .05, .1
were inspected offline on development. The .005 cutoff accepts 64/128 refinements
and preserves 71/128; it is the largest tested cutoff with no development loss.
This gate is a conservative secondary policy, not an established accuracy gain.
A .5 detail fraction receives one 64-case development check. The multiview and
context-view pilots are deferred. Four-megapixel stochastic plumbing receives two
disjoint eight-case chunks per arm; aggregation parameters remain those specified
before those checks. All reserved-validation policies will be frozen explicitly
in a separate JSON artifact before their first evaluation.

## Post-freeze execution and interface additions

For load balancing, the frozen 128-example agreement evaluation is split into two
64-example jobs with offsets 0 and 64, one per worker. Exact source, parameters,
and the total example set remain unchanged; `execution-shards.json` records this
administrative change. MAI's validation baseline runs earlier so its deterministic
first-pass cache is available to both shards.

The user requested unrestricted visual questions and a possible generalist verifier.
The library now exposes `ask_vlm` and an unbenchmarked classification-driven Python
example, plus a proposed Venus/MAI/Qwen pool. These additions do not alter the
running image, frozen policies, or specialist validation. No outcome from the
reserved set is used to tune those policies.
