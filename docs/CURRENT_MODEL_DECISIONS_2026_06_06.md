# Current Model Decisions - 2026-06-06

This document is the handoff-grade decision log for the current Fast-dVLM /
GUI-Owl work. It records what is currently believed, what has been verified,
what remains uncertain, and what must not be claimed yet.

No credentials, server passwords, SSH keys, Hugging Face tokens, GitHub tokens,
or private machine access details belong in this repository.

## 1. Project Objective

The project converts GUI-Owl-1.5-2B-Instruct, a Qwen3-VL style GUI agent, into a
Fast-dVLM / BARD-style block-diffusion VLA.

The model should emit GUI-Owl-compatible mobile tool calls:

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "click",
    "coordinate": [695, 793]
  }
}
```

Swipe target:

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "swipe",
    "coordinate": [658, 888],
    "coordinate2": [745, 361]
  }
}
```

The immediate goal is not RL. The immediate goal is to make the SFT/KD
block-diffusion model stable enough that bd4/bd8/bd16/bd32 decoding can produce:

- strict JSON
- valid `mobile_use`
- the correct action type
- coordinates in the same convention as GUI-Owl-1.5
- nontrivial AndroidWorld task progress

## 2. Canonical Coordinate Convention

The coordinate convention is locked to the GUI-Owl-1.5 / AndroidWorld
convention:

```text
coord_mode: guiowl_norm1000_xy
coordinate order: [x, y]
range: 0..1000
origin: top-left
executor: maps norm1000 -> device pixels
```

Pixel coordinates are not acceptable in final training targets unless they are
explicitly converted to norm1000 first.

Conversion:

```text
x_norm1000 = round(x_pixel / screen_width  * 1000)
y_norm1000 = round(y_pixel / screen_height * 1000)
```

For execution:

```text
x_pixel = round(x_norm1000 / 1000 * screen_width)
y_pixel = round(y_norm1000 / 1000 * screen_height)
```

Why this matters:

- GUI-Owl-1.5 AR output is interpreted as normalized 0..1000.
- AndroidWorld executor expects normalized 0..1000 before mapping to pixels.
- Base bd32 / decode repair paths should follow the same convention.
- Prior Gmail SFT mixed pixel Gmail targets with norm1000 general targets, which
  made AndroidWorld grounding look broken because the model learned a compressed
  pixel-scale coordinate distribution.

This does not prove that every failure is caused by coordinates. It does mean
coordinate validation is a required gate for every future dataset and checkpoint.

## 3. Distillation Teacher Decision

The current resource-aware teacher decision is:

```text
primary teacher: GUI-Owl-1.5-2B / same-model clean AR branch
future teacher: GUI-Owl-1.5-32B-Instruct or 32B-Think, optional later
```

Rationale for 2B self-distillation first:

- same tokenizer
- same model family
- same vision path
- lower alignment risk
- no extra 32B teacher serving cost
- clean AR branch already gives useful mobile action behavior
- the current bottleneck is noisy/diffusion branch stability, not teacher scale

Important caveat:

This does not mean 32B distillation is useless. The 32B Think/Instruct models are
still promising for future reasoning traces, action rationales, and logit
distillation once the bd decoder can produce valid actions. For the current SFT
stability phase, 2B self-distillation is the cheaper and safer first teacher.

## 4. KD Recipe Decision

The old problem:

- Clean/AR branch can remain good while dVLM decoding fails.
- Clean-branch KD alone does not force the noisy branch to learn coherent
  multi-token commits.
- At bd4/bd8/bd16/bd32, tau sweeps did not reliably recover strict JSON or
  action type diversity.

Current KD direction:

```text
L = a * CE_noisy + b * CE_clean + g * KL_noisy
```

Current TPU step-3000 run used:

```text
a = 1.0
b = 0.75
g = 0.25
temperature = 2.0
teacher = same-model clean branch with stop_gradient
bd schedule = 4, 8, 16, 32 equally mixed
```

Why KD is applied to the noisy branch:

- The deployed fast path is the noisy/block-diffusion branch.
- The noisy branch needs a teacher distribution at masked response positions.
- CE_noisy provides hard targets, while KL_noisy gives softer structural and
  local-token mode information.
- KD should augment CE, not replace it.

Risk:

The AR clean branch and noisy branch condition on different contexts. The clean
branch sees a causal true-prefix; the noisy branch sees partially masked block
tokens. Therefore, the teacher target may be aspirational at very high mask
rates. Practical mitigation:

- keep CE_noisy as the main objective
- consider KD warmup or low-noise / late-denoise gating
- consider bd-dependent KD weight, e.g. lower at bd4 and higher at bd32
- monitor strict JSON, action distribution, and coordinate range separately

## 5. Block-Size Scheduling Decision

Do not assume bd32-only training automatically fixes bd4/bd8 deployment.

Current schedule:

```text
bd4  : 25%
bd8  : 25%
bd16 : 25%
bd32 : 25%
```

Reason:

- Each block size creates a different parallel commit distribution.
- bd4/bd8 strict JSON collapse can remain even when bd32 training loss falls.
- The deployment target should be represented during training.

Future variants worth testing:

- deployment-weighted schedule, e.g. bd4/bd8 heavier if final latency target is
  moderate speedup with higher quality
- staged schedule, e.g. bd4 -> bd8 -> bd16 -> bd32
- adaptive schedule based on strict JSON/action-type validation

The schedule should be reported explicitly in every checkpoint README.

## 6. TPU v6e Training Decision

The current TPU target is v6e-4. Key constraints:

```text
devices: 4 TPU devices
HBM: about 32 GiB per device
preferred dtype: bf16
important implementation style: static shapes, prefetch, data parallelism
```

Current step-3000 run metadata:

```text
HF repo: KMK040412/fast-dvlm-guiowl-kd-tpu
HF path: fast-dvlm-kd-tpu/aw-overfit-norm1000-full-pad480-cap96/checkpoint-step003000
source model: KMK040412/ckpt-bard-bd32-gmail-adb-vitlora-e1-final
dataset: KMK040412/aitw-androidworld-overfit-mix
samples: 476,043
batch size: 32
bd schedule: 4/8/16/32, uniform
loss token cap: 96
ctx/pad: 480
vision pad: 96
mRoPE exact: true
DeepStack exact: true
vision_grad: false
ViT LoRA: already merged into the source checkpoint
upload cadence: every 3000 steps
```

Step-3000 train-log snapshot:

```text
first logged loss: 5.4757
step 3000 loss:    2.8271
first ce_noisy:    3.4268
step 3000 ce_noisy:1.6812
first ce_clean:    1.1286
step 3000 ce_clean:0.5681
first kd_noisy:    4.8095
step 3000 kd_noisy:2.8795
last-20 loss mean: 2.2711
last-20 ce_noisy: 1.3299
last-20 ce_clean: 0.5710
last-20 kd_noisy: 2.0518
```

Interpretation:

- Loss is falling.
- Clean CE is low but plateau-like near 0.55-0.60.
- Noisy CE and KD are still much larger, which matches the observed issue:
  diffusion branch remains the hard part.
- This checkpoint is useful for diagnostic eval, not a final paper checkpoint.

## 7. mRoPE + DeepStack Decision

Qwen3-VL / GUI-Owl relies on vision-aware positional handling and DeepStack
visual feature injection. A previous decode path used a weaker legacy setup:

```text
legacy decode: arange position ids + pooler-only vision
training path: get_rope_index mRoPE + visual_pos_masks + DeepStack
```

That mismatch damages spatial grounding, especially for dVLM decoding.

Current local `src/fast_dvlm/decode.py` includes a grounded dVLM decode path that
injects:

- `get_rope_index`-derived vision M-RoPE position ids when available
- `visual_pos_masks`
- `deepstack_visual_embeds`

The grounded path is the correct default for spatial tests. The legacy path is
only useful as an ablation.

## 8. AndroidWorld Status

AndroidWorld is not yet a completed paper-grade benchmark in this repository.

Known state:

- official AndroidWorld package was installed locally in
  `/home/perelman/androidworld_eval_local/.venv`
- model serving uses the separate `internvla_a1` environment
- step-3000 model server health check succeeded on local port 8123
- local emulator setup exists, but old `gmail_pixel34` was unstable for official
  AndroidWorld gRPC runs
- API33 Google APIs x86_64 image was installed as the preferred local emulator
  target
- large-scale task success claims are not established from the local step-3000
  checkpoint

Metric reporting policy:

Always separate:

- strict JSON parse
- strict `mobile_use` validity
- repaired `mobile_use` validity
- repaired count/rate
- task-level success
- action type distribution
- coordinate error/range
- latency

Do not report repaired success as raw strict model success.

## 9. Known Failure Modes

Current blockers:

1. Coordinate mismatch from previous pixel-coordinate Gmail SFT.
2. Noisy branch undertraining relative to clean AR branch.
3. Strict JSON instability at larger block sizes.
4. Action-type collapse, especially swipe overproduction or missing click/type.
5. Large-block parallel commit incoherence.
6. Benchmark/data distribution mismatch: Gmail settings overfit does not prove
   broad AndroidWorld capability.
7. TPU utilization depends heavily on static shapes and input pipeline health.

What tau sweep can and cannot do:

- Tau can trade speed and conservativeness.
- Tau cannot fix a model that never samples valid action structure at bd4/bd8.
- If every tau gives malformed JSON/action collapse, the fix is training-side:
  KD, block curriculum, action balancing, structural token weighting, or steering.

## 10. Steering Module Decision

The steering module remains a planned architecture, not a proven result.

Current design decision:

```text
steering is not denoise-trace-only
steering should read compressed base KV/cache state + recent committed tokens
  + current block denoise trace + current noisy tokens + timestep
steering outputs small gated residuals into selected frozen layers
base model remains frozen
```

The mental model:

```text
frozen Fast-dVLM = compressed GUI action skill library
steering module = residual mode selector / controller
```

This should help choose among existing modes:

- valid JSON vs malformed text
- click vs swipe
- coordinate field vs random number tokens
- tool-call schema vs free-form assistant text

The steering docs are in `docs/STEERABLE_MODULE_PLAN.md`.

## 11. Data Curation Direction

AiTW has already been processed/classified in user-owned Hugging Face assets.
The next broad-corpus curation should not blindly reprocess raw AiTW again.

Immediate useful data direction:

- keep canonical norm1000 target schema
- include source-level coordinate validation
- balance action types
- cap repeated near-duplicate goals
- preserve history/trajectory fields for future steering and reasoning
- avoid benchmark leakage when claiming AndroidWorld generalization

Candidate broad mobile SFT sources:

- AndroidControl-like direct mobile control data
- AMEX-like mobile action/grounding data
- GUI-Odyssey-like long-horizon multi-app trajectories
- OpenMobile-style mobile task data
- static GUI grounding datasets only as auxiliary grounding data, not as direct
  task-success proof

The data curation plan is in `docs/DATA_CURATION_PLAN.md`.

## 12. What Not To Claim

Do not claim:

- AndroidWorld paper-grade benchmark is complete.
- Repaired output equals strict model correctness.
- Coordinate mismatch is the only problem.
- bd32-only training automatically fixes bd4/bd8.
- TPU v6e implementation is fully optimized.
- Steering module has been validated.
- 32B teacher is unnecessary forever.
- Gmail/general overfit proves broad mobile GUI competence.

Safe current claim:

```text
The current canonical target is GUI-Owl-1.5-style normalized 0-1000 mobile_use
output. Large-block Fast-dVLM decoding is not yet a proven AndroidWorld-capable
policy. Current evidence points to coordinate convention mismatch, noisy-branch
undertraining, strict JSON instability, action-type collapse, and block-level
parallel coherence as the main blockers. Repaired outputs are useful for
deployment diagnostics but must be reported separately from strict model
correctness.
```
