# Frozen model interfaces

The default task combines two GUI specialists with a general VLM. All weights are
fixed. The coding agent may change prompts, representations, questions, model
routing, and aggregation in its Python submission; it may not change checkpoints.

- [UI-Venus-1.5-2B](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B) uses Qwen3-VL
  and has an official vLLM recipe and point-grounding adapter.
- [MAI-UI-2B](https://huggingface.co/Tongyi-MAI/MAI-UI-2B) uses Qwen3-VL and provides
  grounding reasoning followed by a structured answer.
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) supplies general image/text
  reasoning as well as native grounding. Use it for any questions the program
  considers useful, including classification and verification.

The separate [Jedi-3B-1080p](https://huggingface.co/xlangai/Jedi-3B-1080p) variant
uses Qwen2.5-VL, a computer-use schema, and resized pixel coordinates. It remains an
available three-specialist configuration. A configuration change defines a new
comparison; do not mix outcomes from different pools.

Other small specialists considered for future adapters include
[GUI-AIMA-3B](https://github.com/sjz5202/GUI-AIMA),
[GUI-Actor-3B](https://github.com/microsoft/GUI-Actor),
[Phi-Ground-4B](https://huggingface.co/microsoft/Phi-Ground), and
[ShowUI-2B](https://huggingface.co/showlab/ShowUI-2B).
GUI-AIMA and GUI-Actor use custom attention/pointer inference; their outputs should
not be assumed to work through an ordinary text-completion server. Phi-Ground
requires a strict input layout and an explicit letterbox mapping. These are not
part of the default task or validated interfaces.
