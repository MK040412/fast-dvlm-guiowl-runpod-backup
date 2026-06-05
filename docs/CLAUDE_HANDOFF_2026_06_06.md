# Claude Handoff - 2026-06-06

This is a concise operational handoff for another agent continuing the project.

## Do Not Do

- Do not commit tokens, passwords, SSH keys, server IPs, or private machine
  credentials.
- Do not put model weights, `.safetensors`, parquet datasets, AVD images, mp4s,
  screenshots, or large logs in GitHub.
- Do not report repaired metrics as strict model correctness.
- Do not change the coordinate convention away from GUI-Owl norm1000 unless the
  entire executor/training stack is deliberately changed.

## Current Canonical Decisions

1. Coordinates:

```text
guiowl_norm1000_xy, [x, y], 0..1000, top-left origin
```

2. Teacher:

```text
use GUI-Owl-1.5-2B / same-model clean AR branch first
32B Think/Instruct is future work
```

3. KD:

```text
train noisy branch directly with CE_noisy + KL_noisy
keep CE_clean as an anchor
do not rely on clean-only KD
```

4. Block schedule:

```text
include bd4/bd8/bd16/bd32 during training
validate each block size separately
```

5. Vision handling:

```text
mRoPE + DeepStack must be on in training and grounded decode
legacy arange/pooler-only decode is an ablation, not the main path
```

6. Steering:

```text
planned, not proven
KV/cache-aware residual controller
base frozen
```

## Current Step-3000 Artifact

```text
HF repo: KMK040412/fast-dvlm-guiowl-kd-tpu
HF path: fast-dvlm-kd-tpu/aw-overfit-norm1000-full-pad480-cap96/checkpoint-step003000
source model: KMK040412/ckpt-bard-bd32-gmail-adb-vitlora-e1-final
dataset: KMK040412/aitw-androidworld-overfit-mix
```

The checkpoint includes:

- `model.safetensors`
- `train_log.jsonl`
- `run_config.json`
- `data_summary.json`
- `checkpoint_manifest.json`
- `jax_export_summary.json`
- `vit_lora_merged_config.json`
- `tpu_usage.jsonl`

Step-3000 loss snapshot:

```text
loss:     2.8271
ce_noisy: 1.6812
ce_clean: 0.5681
kd_noisy: 2.8795
```

Interpretation:

- useful diagnostic checkpoint
- not final
- noisy branch remains the hard part
- validate coordinates/action/strict JSON before deciding resume vs restart

## Local Eval Status

Local server can load the step-3000 checkpoint. The model server health check
passed for:

```text
mode=dvlm
bd=32
tau=0.9
grounded=1
port=8123
```

Official AndroidWorld runner environment exists, but local emulator stability
needs finishing:

```text
runner env: /home/perelman/androidworld_eval_local/.venv
model env:  /home/perelman/miniconda3/envs/internvla_a1
old AVD:    gmail_pixel34, unstable
preferred:  API33 Google APIs x86_64 AVD
```

## Immediate Next Technical Checks

1. Evaluate step-3000 on a small local smoke suite:

```text
bd1, bd4, bd8, bd16, bd32
tau sweep if needed
repair on/off separated
```

2. Check coordinate range:

```text
Are predicted coordinates spread over 0..1000?
Or are they still compressed to pixel-like ranges?
```

3. Check action distribution:

```text
click/swipe/type/system_button/terminate counts
parse failure count
```

4. If coordinates are still pixel-compressed:

```text
restart from cleaner bd32 base with norm1000 data
```

5. If coordinates are corrected but JSON/action fails:

```text
resume with stronger noisy KD, block-size validation, action-balanced sampling
```

## Relevant Docs

- `docs/CURRENT_MODEL_DECISIONS_2026_06_06.md`
- `docs/COORDINATE_CONVENTION.md`
- `docs/TPU_KD_TRAINING_NOTES.md`
- `docs/ANDROIDWORLD_LOCAL_STATUS.md`
- `docs/DATA_CURATION_PLAN.md`
- `docs/STEERABLE_MODULE_PLAN.md`
- `docs/KNOWN_FAILURES_AND_NON_CLAIMS.md`
