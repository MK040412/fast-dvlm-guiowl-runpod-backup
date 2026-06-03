# Measured results (NVIDIA L40S 46GB, GUI-Owl-1.5-2B / Qwen3-VL, bf16)

All numbers below were measured on a single L40S during development. H100 numbers are
projections (see README); the code paths are identical.

## Pipeline validation
| Stage | Result |
|---|---|
| Teacher AR inference (1 AITW screenshot) | Coherent GUI action: `"Click on the Chrome browser icon in the dock."` (peak 4.4GB) |
| Dual-stream mask (N2N/N2C/C2C) | Pattern verified; `tests/test_mask.py` PASS |
| Dual-stream forward (4D mask passthrough) | logits finite; mask-sensitivity 3.44 vs causal (mask is applied) |
| 20-step training smoke | loss 9.19 -> 2.13 DOWN, no OOM, peak 20.7GB |

## Full-episode overfit (5 episodes, 300 steps, frozen+cached vision)
- loss **5.74 -> ~0.0**; throughput **1.91 it/s, 2805 tok/s**, peak VRAM **18.7GB**.
- Output verification (after overfit):
  - **AR (causal)** reproduces targets exactly: `click(x=0.709, y=0.776)`, `swipe(...)`, `press_home()`.
  - **dVLM (block-diffusion fill)** of masked last action: `status(complete)` correct.

## Throughput levers (controlled A/B, same episodes)
| Lever | Effect |
|---|---|
| FlexAttention vs dense sdpa (block-sparse mask) | **2.03x** @ L=2048, **2.51x** @ L=4096 (grows with length), ~1.3x less mem |
| grad-ckpt OFF + fused AdamW | **1.21x** (629 -> 520 ms/step; 19.0 -> 34.4GB) |
| frozen + cached vision vs trainable+recompute | **1.26x** (829 -> 656 ms/step; -1.7GB; full-FT is only +26%) |

## Single-device, batch=1 inference: dVLM vs AR (overfit model)
| | tokens | NFE | latency |
|---|---|---|---|
| AR | 17.5 | 17.5 | 467 ms (37.5 tok/s) |
| dVLM | 17.5 | **6.0** | 168 ms (104 tok/s) |
- **tokens/NFE = 2.92x** (forwards saved = realisable single-user latency speedup with a KV-cache).
- Raw wall-clock **2.78x** faster even without a KV-cache on dVLM. (Overfit -> optimistic;
  paper reports ~1.95-2.63 tok/NFE on real benchmarks, up to 6.18x with SGLang + FP8 serving.)
