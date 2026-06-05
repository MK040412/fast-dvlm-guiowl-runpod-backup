# TPU KD Training Notes

This document records the current TPU v6e training recipe and the reasoning
behind the KD changes.

## Hardware Assumption

Current TPU target:

```text
TPU type: v6e-4
devices: 4
HBM: about 32 GiB per device
dtype: bf16
```

This is not an H100. The training path should favor:

- static shapes
- fixed padding
- bf16
- prefetching
- data-parallel sharding
- avoiding Python stalls
- minimizing dynamic image-grid retracing

## Current Step-3000 Checkpoint

Artifact:

```text
HF repo: KMK040412/fast-dvlm-guiowl-kd-tpu
HF path: fast-dvlm-kd-tpu/aw-overfit-norm1000-full-pad480-cap96/checkpoint-step003000
```

Local diagnostic copy:

```text
/home/perelman/local_gmail_eval/models/ckpt_kd_aw_step3000/fast-dvlm-kd-tpu/aw-overfit-norm1000-full-pad480-cap96/checkpoint-step003000
```

Checkpoint metadata:

```text
source model: KMK040412/ckpt-bard-bd32-gmail-adb-vitlora-e1-final
step: 3000
final: false
format: safetensors
jax_export_step: 3000
tensor count: 626
mRoPE exact: true
DeepStack exact: true
```

Training metadata:

```text
dataset: KMK040412/aitw-androidworld-overfit-mix
rows: 476,043
parquet shards: 14
batch size: 32
epochs: 1 target
bd schedule: 4/8/16/32, uniform
ctx_cap: 480
pad_to: 480
noisy_pad_to: 480
vision_pad_to: 96
loss_token_cap: 96
vision_grad: false
vision precompute batch size: 16
optimizer: adamw_bf16
lr: 1e-6 in the current run
```

Loss weights:

```text
CE_noisy weight: 1.0
CE_clean weight: 0.75
KD_noisy weight: 0.25
KD temperature: 2.0
teacher: same-model clean branch with stop_gradient
```

## Step-3000 Loss Snapshot

From `train_log.jsonl`:

```text
first logged loss:       5.4757
step 3000 loss:          2.8271
minimum logged loss:     1.5935
last-20 loss mean:       2.2711

first CE_noisy:          3.4268
step 3000 CE_noisy:      1.6812
last-20 CE_noisy mean:   1.3299

first CE_clean:          1.1286
step 3000 CE_clean:      0.5681
last-20 CE_clean mean:   0.5710

first KD_noisy:          4.8095
step 3000 KD_noisy:      2.8795
last-20 KD_noisy mean:   2.0518
```

Interpretation:

- Training is reducing loss.
- Clean branch is already much easier than noisy branch.
- Noisy CE/KD remain the hard terms.
- This supports the current diagnosis: the deployed dVLM branch needs more
  direct supervision and validation.

## Why Noisy-Branch KD

Problem:

```text
clean AR branch can be valid and grounded
large-block dVLM branch can still output malformed JSON/action collapse
```

Therefore KD must touch the branch used by fast decoding.

Current loss form:

```text
L = a * CE_noisy + b * CE_clean + g * KL_noisy
```

KD target:

```text
teacher logits = clean AR branch, stop_gradient
student logits = noisy diffusion branch at masked response positions
```

Indexing rule:

- Decode uses token shift: token `i` is predicted from hidden position `i-1`.
- KD and CE must align teacher next-token logits to the student shifted logits.
- Off-by-one errors here can train the model to produce plausible but misplaced
  structure tokens.

## Why Mixed Block Sizes

bd32-only training is not guaranteed to fix bd4/bd8/bd16 behavior.

The current schedule:

```text
bd4  : 25%
bd8  : 25%
bd16 : 25%
bd32 : 25%
```

Rationale:

- each block size has a different denoise/parallel-commit distribution
- bd4/bd8 are likely practical deployment targets if bd32 quality is too low
- validation should report each block size separately

## Current Concerns

1. Learning rate may be conservative.

The current run uses `lr=1e-6`. For a future run or resume, consider:

```text
warmup: 100-300 steps
peak lr: about 1e-5, to be validated
schedule: cosine or linear decay to 1e-6
grad accumulation: 4-8 if TPU memory allows
weight decay: about 0.1 for a more standard SFT recipe
```

2. Clean branch may preserve old source-model behavior.

The current source model includes the previous Gmail/Vision LoRA history. If the
step-3000 eval shows coordinate distributions remain pixel-compressed, consider
restarting from a cleaner bd32 base with the corrected norm1000 data.

3. TPU utilization is implementation-limited.

Low utilization does not mean v6e is inherently unsuitable. It usually points to:

- input pipeline stalls
- dynamic shapes
- small effective batch
- expensive Python-side preprocessing
- insufficient compile/fuse friendliness

## Required Eval After Every Checkpoint

Every uploaded checkpoint should be evaluated for:

- strict JSON rate
- strict `mobile_use` rate
- repaired `mobile_use` rate
- repaired count/rate
- action-type distribution
- coordinate range and L2 error
- task-level success on small AndroidWorld smoke tasks
- latency and NFE
- per-bd metrics for bd1/bd4/bd8/bd16/bd32

Do not use scalar training loss alone as evidence of AndroidWorld readiness.
