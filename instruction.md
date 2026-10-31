# Improve the Python GUI-grounding harness

You are the optimization agent. Edit `submission.py`, an inference program that
maps a screenshot and instruction to a normalized point `[x,y]` or refusal. Use
Python to design any image representations, sampling, model routing, verification,
aggregation, and stopping rules that help. The VLM checkpoints are frozen inference
services; they do not write or execute Python. The submitted file is the algorithm.

1. Read development results and state a falsifiable hypothesis.
2. Edit the Python inference program and record the code hash and rationale.
3. Evaluate matched visible examples under the fixed model-call/token budget.
4. Compare accuracy, errors, calls, tokens, and time; keep or revert the change.
5. Freeze candidates before reserved validation. Preserve negative trials.

Use model-specific prompts and coordinate frames from `docs/model-formats.md`.
All requests share one H100 and are serialized. Failed calls count; no hidden
retries. Matched study policies use at most six requests per example and bounded
outputs. Models, serving settings, and evaluation examples remain fixed within a
comparison. Report checkpoint replacement separately from harness improvements.

Policies receive only image pixels and instruction. Do not retrieve annotations,
hard-code answers, or use benchmark identifiers as prediction features. OSWorld-G
is reserved for a later transfer evaluation. The current evaluator is a trusted
research harness, not an adversarially isolated benchmark judge.
