# Fast-dVLM × GUI-Owl  —  block-diffusion conversion of a Qwen3-VL GUI agent

Convert **GUI-Owl-1.5-2B** (`mPLUG/GUI-Owl-1.5-2B-Instruct`, a **Qwen3-VL** model) from an
autoregressive VLM into a **block-diffusion VLM** via Fast-dVLM's *direct conversion* recipe
(arXiv **2604.06832**, NVlabs/Fast-dLLM `fast_dvlm`), trained on **AITW-General**.

Block-diffusion decoding emits **multiple tokens per forward pass**, so a single-user request
finishes in far fewer sequential forwards than autoregressive decoding — the latency win is
`tokens / NFE` (see [`docs/RESULTS.md`](docs/RESULTS.md)).

> **Status.** The full pipeline (dual-stream mask, dual-CE loss, block annealing,
> auto-truncation, complementary pair, AR + diffusion decode) is **implemented and verified on
> an L40S**. The H100 fast paths (`--attn flex`, `--compile`, `--no-grad-ckpt`, large
> `--grad-accum`) are wired and individually micro-benchmarked, but the full H100 config should
> be smoke-tested once on the target box. Two fidelity items are deliberately deferred (full
> vision mRoPE + DeepStack injection) — see *Faithfulness* below.

## What's faithful to the paper
- **Dual-stream input** `[w_t ; x]`: noisy stream = masked response tokens only (vision dropped,
  "vision-efficient concat"); clean stream = vision + full text.
- **Dual-stream mask** — `N2N` (bidirectional within a block), `N2C` (noisy → clean of earlier
  turns), `C2C` (clean is causal). Multi-turn: each step's action is its own response block.
- **Dual CE loss** `L = 0.5·CE_noisy(2 pairs) + CE_clean(pair-1)` with HF token-shift (= α=β=0.5),
  complementary masking pair, per-block Bernoulli noising.
- **Block-size annealing** `{2,4,8,16,32}` (√-progress schedule) + **auto-truncation** of each
  response's last block at the turn boundary.

## Install
```bash
pip install -r requirements.txt
# datasets must be installed WITHOUT deps (its fsspec ceiling conflicts with the pinned stack):
pip install "datasets" --no-deps
# H100 extras (optional, see "H100 optimisation"):
pip install flash-attn --no-build-isolation bitsandbytes
```
Pinned because it matters: `torch 2.7.1+cu126 / triton 3.3.1 / transformers 5.9.0`. `flex`
attention needs `torch >= 2.5` (have 2.7.1).

## Data
```bash
HF_TOKEN=hf_xxx python scripts/download_aitw.py --split train --start 0 --n 1 --out ./data
```
AITW-General `standard` = 256 train + 32 test parquet shards (~100MB each, ~28GB total — pull
shard-by-shard). Images are **raw RGB** (W=540, height by device; `CUSTOM_DEVICE` is filtered).

## Quickstart
```bash
# L40S — verified config (sdpa, grad-ckpt, frozen vision), fast overfit demo:
python train.py --config configs/l40s.yaml --data ./data/standard/train-00000-of-00256.parquet --overfit

# H100 SXM — maximise batch + throughput:
python train.py --config configs/h100.yaml --data './data/standard/train-*.parquet' --save ./ckpt

# verify the converted model (AR vs dVLM decoding, single device, batch=1):
python scripts/decode_demo.py --model ./ckpt --data ./data/standard/train-00000-of-00256.parquet
python tests/test_mask.py        # mask unit test, no GPU
```

## H100 SXM optimisation (max batch + throughput)
H100 SXM (80GB HBM3, ~3.35 TB/s, FP8 Tensor Cores) vs the L40S (46GB, ~0.86 TB/s) we developed
on. `configs/h100.yaml` turns every lever on. Measured speedups (L40S A/B, see RESULTS):

| Lever | Flag | Measured (L40S) | Why it helps more on H100 |
|---|---|---|---|
| **FlexAttention block-sparse mask** | `--attn flex` | **2.0–2.5×** attn (grows with len) | the dual-stream mask is ~33% dense; flex skips empty blocks + fuses, and the win grows with the longer contexts H100 can hold |
| **grad-ckpt off + fused AdamW** | `--no-grad-ckpt --optim adamw_fused` | **1.21×** | 80GB removes the memory pressure that forced checkpointing |
| **frozen + cached vision** | `--freeze-vision` (default) | **1.26×** + 1.7GB | no per-step ViT; embeds cached across epochs |
| **bigger effective batch** | `--grad-accum N` | — | L40S sat at batch≈1; H100 fits much more |
| **torch.compile** | `--compile` | — (required for flex) | kernel fusion; H100 benefits most |
| **fewer vision tokens** | `--max-pixels` | — | shortens the O(L²) attention that dominates |

These compound. A reasonable H100 target is **several× the L40S throughput** at a much larger
effective batch — confirm with a short run on the box.

**About "flash attention" specifically:** vanilla **FlashAttention-2/3** only supports
causal / sliding-window masks, *not* our arbitrary dual-stream mask, so we use
**FlexAttention** (`torch.nn.attention.flex_attention`) — the flash-style fused, block-sparse
kernel that *does* take an arbitrary `BlockMask`. This is exactly what the Fast-dVLM reference
uses. (Install `flash-attn` only if you also want FA2/3 for a plain-causal eval path.)

**Recommended next levers (not auto-enabled):**
- **Sequence packing** — concat episodes into `ctx_cap` buffers with a document-aware flex
  `mask_mod` (combine doc-isolation with the dual-stream rule) → near-100% GPU utilisation. The
  single-episode + `--grad-accum` path ships today; packing is the highest-value follow-up.
- **FP8** matmuls via NVIDIA TransformerEngine → ~2× matmul throughput on H100.
- **Multi-GPU** (SXM NVLink): `torchrun` + FSDP → near-linear scaling.
- **8-bit AdamW** (`--optim adamw_8bit`) to trade a little speed for a much larger batch.

## Faithfulness — deferred items (flagged in `forward.py`)
1. **position_ids = arange standard RoPE** instead of full vision mRoPE (`get_rope_index` needs
   `mm_token_type_ids`). Affects spatial grounding quality, not pipeline correctness.
2. **DeepStack injection skipped** (embedding-level vision merge only). The Qwen3-VL text model
   already accepts `visual_pos_masks` + `deepstack_visual_embeds`; add them on the clean half.

## Repo layout
```
train.py                     # CLI training entrypoint (all levers as flags)
src/fast_dvlm/
  mask.py                    # dual-stream mask: dense (sdpa) + BlockMask (flex)
  flex_attn.py               # registers the FlexAttention interface for transformers
  forward.py                 # dual-stream multimodal forward + dual CE loss
  decode.py                  # AR vs dVLM block-diffusion decoding
  data.py                    # AITW loader, raw-RGB decode, episode build, vision cache
configs/{h100,l40s}.yaml     # presets
scripts/{download_aitw,decode_demo}.py
tests/test_mask.py           # mask unit test (no GPU)
docs/{PORT_SPEC,RESULTS}.md  # line-by-line port spec + measured numbers
```

## References
- Fast-dVLM: *Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM*, arXiv 2604.06832
- Fast-dLLM v2: *Efficient Block-Diffusion LLM*, arXiv 2509.26328 · code: github.com/NVlabs/Fast-dLLM
- GUI-Owl-1.5-2B: `mPLUG/GUI-Owl-1.5-2B-Instruct` (Qwen3-VL) · Data: `cjfcsjt/AITW_General`

*Built and verified on RunPod L40S; packaged for H100 SXM.*
