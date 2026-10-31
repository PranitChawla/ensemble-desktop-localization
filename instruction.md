# Improve Python-tool-assisted desktop GUI grounding

Develop the inference policy in `submission.py`. Given an original screenshot and
a natural-language instruction, return a normalized point `[x,y]` or a refusal.
You have a persistent arbitrary Python tool and three frozen VLMs. Invent whatever
computation, model orchestration, representations, or verification procedures are
useful within the fixed budget.

The initial controller is Qwen, but the policy may use any member of the pool.
Python programs are generated dynamically at inference time. Their variables and
working files persist within an example and reset between examples.

## Research loop

1. Read prior visible results and state one falsifiable hypothesis.
2. Change `submission.py`; keep a short record of the change and its rationale.
3. Run the same visible development examples with the fixed serving configuration
   and model-call budget. Use a fresh output directory for each experiment.
4. Compare accuracy, operational failures, model calls, prompt/output tokens, and
   elapsed time. Keep or revert the policy based on the evidence.
5. Preserve the best candidate and its exact code hash, model revisions, and logs.

Initially use the synthetic screen and the three-example visible integration sample
to debug plumbing. They are not a meaningful benchmark. Build screenshot-disjoint
development/validation manifests before hill climbing on ScreenSpot-Pro.

## Fixed comparison contract

- Freeze model commits, processor/serving settings, datasets, scorer, and budgets
  for matched comparisons. Infrastructure changes require a new matched baseline.
- The initial budget is 12 model requests total per example, including controller
  calls, helper calls, malformed outputs, and failed requests. There are no hidden
  retries. Each generation is capped at 1,024 output tokens.
- All requests share one H100 and are serialized. Python runs on CPU initially;
  tool steps have a wall-time limit. The outer Modal experiment also has a timeout.
- Python sees the screenshot and instruction, not evaluator labels. Do not retrieve
  benchmark annotations, hard-code answers, or use benchmark identifiers as features.
- OSWorld-G stays reserved and must not be downloaded, inspected, or queried during
  visible development. Its eventual final protocol must be defined separately.

This is a trusted initial research harness. The arbitrary Python interpreter shares
the container filesystem and network; it is not yet a hardened OpenRSI Judge. Do
not claim adversarial evaluation isolation or independent hidden-test secrecy.

