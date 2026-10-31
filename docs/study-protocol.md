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

