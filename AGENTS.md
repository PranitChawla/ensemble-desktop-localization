# Working in this repository

For a benchmark improvement attempt, read `instruction.md` and `task.toml` first.
The editable submission is `submission.py`. The VLM checkpoints and task/evaluator
files are fixed during that attempt. General `ask_vlm` questions are allowed;
VLM responses are evidence, never executable Python.

For task infrastructure maintenance explicitly requested by the user, update the
relevant harness files and run the CPU checks in `.github/workflows/tests.yml`.
Local tests must not start cloud GPUs. Do not claim GPU validation unless the
current configuration was actually tested. Keep experiment output and analysis
under ignored local `outputs/` or `work/`, not in the tracked task repository.
