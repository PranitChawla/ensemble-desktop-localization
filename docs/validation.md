# Initial H100 validation — 2026-10-31

The initial format-aware pipeline passed on one **NVIDIA H100 80GB HBM3** in the
configured Modal account. The app and GPU function were
both named **`ensemble-desktop-localization`**.

- [Modal run](https://modal.com)
- [Full report](validation/h100-20261031T215032Z.json)
- Candidate SHA256: `61892a4835bf94d82b495f6f4f1a56c141827cf5e28ee488ca9d99213405db1f`

## Serving and memory

All three pinned models ran simultaneously as separate loopback vLLM servers, each
with a 0.25 memory fraction, 16,384-position maximum context, one active sequence,
BF16 weights/cache defaults, and eager execution. Requests were serialized.

| Stage | Device memory used | Device memory free |
| --- | ---: | ---: |
| Qwen ready | 20,719 MiB | 60,353 MiB |
| Qwen + Ministral ready | 41,277 MiB | 39,795 MiB |
| All three ready | 62,046 MiB | 19,026 MiB |
| Highest sampled usage / lowest sampled free | **62,362 MiB (60.9 GiB)** | **18,710 MiB (18.3 GiB)** |

Memory was sampled once per second, with 169 samples. This is a sampled device-wide
peak, not an exact allocator trace. No OOM occurred in the tested workload.

## Exercised behavior

- Image requests to all three servers using a 1920×1080 synthetic desktop.
- Long text requests with actual prompt lengths of **10,707 tokens (Qwen)**,
  **10,249 (Ministral)**, and **7,665 (LFM)**.
- Persistent Python state, a real Python-initiated LFM image request, and returning
  an intermediate image to the controller interface.
- Grounding requests with the documented Qwen 0–1000 convention and explicitly
  prompted pixel/0–1 protocols for Ministral/LFM, with raw and converted results.
- A full agent run that returned Qwen's original `[787,731]` point on a 0–1000 grid,
  converted it to `[0.787,0.731]`, and hit the synthetic Save button.
- Three visible ScreenSpot-Pro VS Code examples: **1/3 correct**, zero operational
  failures, three model calls, 3.84 seconds measured evaluation time.

The three public examples are an integration sample, not a representative benchmark.
The baseline remains unoptimized. On the single synthetic fixture, Qwen localized
the target correctly; Ministral and LFM did not. LFM did correctly read the two
button labels through the Python helper. These observations do not establish that
the pool improves over Qwen alone.

The text-context and image probes were separate requests. This run does not prove
that every maximum-size multi-image conversation fits a full 16K context. Test the
intended worst-case multimodal history before increasing concurrency or adding a
fourth server. LFM produced valid text but did not follow the requested READY-only
answer on its long-context probe.

## Reproducibility and lifecycle

Runtime: vLLM 0.32.0, Transformers 5.19.0, Hugging Face Hub 2.2.0, Pillow 12.3.0,
NumPy 2.2.6, HTTPX 0.28.1. Model/data revisions and further runtime versions are in
the full report. All **25 CPU tests** and GitHub Actions checks passed.

Reports and full server/request logs are in the environment's
`ensemble-desktop-localization-results-v1` volume under `/h100-20261031T215032Z/`.
The run completed and its GPU container stopped. No public endpoint or always-on
GPU deployment was created. OSWorld-G was not downloaded or evaluated.

