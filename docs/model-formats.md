# Grounding format audit

Reviewed against publisher documentation and code for the exact checkpoints in
`models.json` and the optional Jedi variant. Coordinate units are explicit;
full parameter counts include vision encoders. The task imposes no question taxonomy.

| Alias | Exact checkpoint | Full parameters | Native coordinate frame |
| --- | --- | ---: | --- |
| `qwen` | Qwen3.5-2B, `15852e8c16360a2fea060d615a32b45270f8a8fc` | ~2.27B | 0–1000 relative to supplied image |
| `venus` | UI-Venus-1.5-2B, `bc4c6e62a1c545e043761439be95646d801853a4` | 2,438,696,960 | 0–1000 relative to supplied image |
| `jedi` (variant) | Jedi-3B-1080p, `4aef56c6b08a98b534a32abb8aaaa35a1571a32d` | 3,754,622,976 | Absolute pixels in processor-resized image |
| `mai` | MAI-UI-2B, `503050934809558c8dfd2ddedaf9621fa74ac2de` | 2,127,532,032 | Released grounding client divides answer coordinates by **999** |

### Venus

Read the [pinned model card](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B/blob/bc4c6e62a1c545e043761439be95646d801853a4/README.md),
config and processor settings, the [official v1.5 grounding implementation](https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/models/grounding/ui_venus1_5_gd.py),
and [technical report §3.1.1](https://arxiv.org/html/2602.09082v1).
The implementation uses an image-first user message and a center-point prompt,
parses `[x,y]`, divides by 1000, and recognizes `[-1,-1]` as refusal. The adapter
preserves that protocol and maps refusal to `None`. It does not execute navigation.

### Jedi

Read the [pinned card and executable example](https://huggingface.co/xlangai/Jedi-3B-1080p/blob/4aef56c6b08a98b534a32abb8aaaa35a1571a32d/README.md),
processor configuration (`patch_size=14`, `merge_size=2`, `max_pixels=2116800`),
[official demo](https://github.com/xlang-ai/OSWorld-G/blob/main/demo.py), and
[computer-use schema](https://github.com/xlang-ai/OSWorld-G/blob/main/agent_function_call.py).
The demo declares the resized display width/height in the tool description and
converts returned pixel coordinates by original/resized dimensions. The harness
explicitly performs the aligned resize, supplies the computer-use schema, parses
`computer_use.arguments.coordinate`, and maps back to the caller's original frame.
The tool-call output is a coordinate representation; no computer action is executed.
The model card's visualization omits a conversion that is present in the repository
demo; the adapter follows the demo's explicit resized-coordinate mapping.

### MAI

Read the [pinned card](https://huggingface.co/Tongyi-MAI/MAI-UI-2B/blob/503050934809558c8dfd2ddedaf9621fa74ac2de/README.md),
config/template/processor, the [official grounding client](https://github.com/Tongyi-MAI/MAI-UI/blob/main/MAI-UI/src/mai_grounding_agent.py),
[system prompt](https://github.com/Tongyi-MAI/MAI-UI/blob/main/MAI-UI/src/prompt.py),
and [older evaluation adapter](https://github.com/Tongyi-MAI/MAI-UI/blob/main/MAI-UI/evaluation/grounding/models/MAI_UI.py).
The released client uses `SCALE_FACTOR = 999` and extracts only the JSON inside
`<answer>`; the older batch evaluator divides by 1000, while another path divides
by resized dimensions. These sources conflict. Our declared `norm999` protocol
follows the released grounding client and does not guess units from output size.
The answer region is parsed separately from `<grounding_think>` to avoid treating
reasoning coordinates as additional predictions. Generation is bounded; truncated
answers count as failures. No retries from the publisher's client are copied.

The publisher prompts and Jedi schema are adapted from the Apache-2.0 official
repositories cited above; see `THIRD_PARTY_NOTICES.md`.

## Qwen

The pinned [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md), configuration, and template were reviewed alongside the
[Qwen team's accepted coordinate explanation](https://github.com/QwenLM/Qwen3.8/discussions/56)
and [official 2D grounding notebook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb).
The documented convention is 0–1000 relative coordinates, x then y, with xyxy boxes.
The notebook maps points by width/1000 and height/1000. The repository containing
the discussion was renamed; the discussion explicitly concerns Qwen3.5.

## Frame mapping

`grounding/coordinates.py` supports declared `pixels`, `norm1000`, `norm999`, and
`norm01` units. It rejects ambiguous multiple targets, non-finite values, booleans,
and out-of-bounds points. It never guesses units or clips coordinates.
`ImageFrame` supplies explicit scale, translation, or affine mappings to the original
screenshot. An image without a known frame does not acquire an invented original
coordinate. Nonlinear transformations need caller-owned inverse mapping.
General `ask_vlm` returns text and never silently interprets that text as coordinates.
