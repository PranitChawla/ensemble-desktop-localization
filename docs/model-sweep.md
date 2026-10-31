# Small GUI-model literature and implementation sweep

Reviewed 2026-10-31. The optimization agent edits a Python harness around frozen
VLMs. The VLMs are inference services, not Python-tool-using optimization agents.
Published scores below describe the authors' protocols, not our reproduction.

| Candidate | Backbone / output mechanism | Evidence and practical decision |
| --- | --- | --- |
| UI-Venus-1.5-2B | Qwen3-VL; relative center point | [Official card](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B), [report](https://arxiv.org/abs/2602.09082): author-reported 57.7% ScreenSpot-Pro for **2B**, standard vLLM recipe. Selected. |
| MAI-UI-2B | Qwen3-VL; reasoning plus answer coordinate | [Official project](https://github.com/Tongyi-MAI/MAI-UI), [checkpoint](https://huggingface.co/Tongyi-MAI/MAI-UI-2B): project reports **57.4% for 2B**. Its card's 73.5% family-level result must not be attributed to 2B. Selected; strict prompt and coordinate audit required. |
| Jedi-3B-1080p | Qwen2.5-VL; computer-use coordinate | [Official card/demo](https://huggingface.co/xlangai/Jedi-3B-1080p), [paper](https://arxiv.org/abs/2505.13227): task-specific synthesis and a distinct training recipe; standard vLLM, but resized pixel coordinates. Selected. |
| GUI-AIMA-3B | Qwen2.5-VL; context-anchor attention grounding | [Official code](https://github.com/sjz5202/GUI-AIMA): strong reported results, but attention extraction/custom inference differs from a standard text-completion endpoint. The [official example](https://github.com/sjz5202/GUI-AIMA/blob/main/eval/example_inference.py) loads `Qwen2_5_VLForConditionalGenerationWithPointer`; [inference](https://github.com/sjz5202/GUI-AIMA/blob/main/src/gui_aima/inference.py) extracts attention and pointer-head scores. Strong follow-up, requiring an adapted service. |
| GUI-Actor-3B | Qwen2.5-VL; learned attention/pointer grounding | [Official code](https://github.com/microsoft/GUI-Actor) reports 42.2% without verifier and 45.9% with verifier. Custom grounding output requires an adapted service rather than assuming generic chat is sufficient. Deferred. |
| Phi-Ground-4B-7C | Phi-3.5-Vision; bounding box | [Official card](https://huggingface.co/microsoft/Phi-Ground): independent backbone, fixed 1008×672 letterboxed input, strict prompt, custom code and linked vLLM implementation. Attractive diversity follow-up, but adds integration risk in a bounded study. Deferred. |
| ShowUI-2B | Qwen2-VL; visual action model | [Official checkpoint](https://huggingface.co/showlab/ShowUI-2B), [paper](https://arxiv.org/abs/2411.17465): useful small baseline, older recipe and custom visual-token handling. Its reported ScreenSpot result is not a ScreenSpot-Pro score. Deferred. |

Checkpoint specialization takes priority over backbone diversity, per the updated
study requirement. Jedi and the two Qwen3-VL specialists may still produce different
errors because of data, objectives, prompts, and output representations. Whether
those differences help a real ensemble must be measured; architecture labels alone
do not establish complementarity.

The initial Qwen3.5/Ministral/LFM pool is retained as a historical comparison. Its
Ministral and LFM results under the tested prompted coordinate protocols were too
weak to justify keeping them resident. A pool replacement is reported separately
from an improvement caused by a Python harness change.

## Generalist verifier alternative

A pool with three coordinate specialists may lack a strong general visual reasoner.
The existing selection tests already use general classification questions through
`pool.ask`/`pool.chat`; the public `ask_vlm` API now makes that capability explicit,
with arbitrary questions, multiple Python-created images, optional JSON schemas,
and the same shared request budget. Model outputs remain data; the coding agent
writes the Python decision tree.

`configs/generalist-verifier.json` proposes replacing Jedi with the pinned
Qwen3.5-2B checkpoint already audited in `configs/generalists.json`, retaining Venus
and MAI. This is a **proposed, unbenchmarked pool**. Its original one-megapixel Qwen
limit is retained; the earlier weak grounding score does not establish whether it
is a useful classifier/verifier. The checkpoint can also return native 0–1000
coordinates. The frozen specialist validation is not mixed with this proposal.

`candidates/classify_then_ground.py` demonstrates target classification, conditional
expert queries, Python candidate annotation, and disagreement verification in at
most four calls. It is an **unbenchmarked API example**, not the selected policy.
The next controlled comparison should hold the images, call budget, and decision
tree fixed while substituting a generalist verifier for the specialist verifier.
