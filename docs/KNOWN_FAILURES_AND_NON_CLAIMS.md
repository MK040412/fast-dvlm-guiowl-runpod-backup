# Known Failures and Non-Claims

This file exists to prevent overclaiming.

## Known Failure Modes

### 1. Coordinate Convention Mismatch

Some prior Gmail SFT data used pixel coordinates while the model/executor stack
expects GUI-Owl normalized 0..1000 coordinates.

Impact:

- coordinates can look compressed toward the upper-left under AndroidWorld
  execution
- training loss can decrease while task-level grounding remains wrong

Required fix:

- validate and convert every source to `guiowl_norm1000_xy`

### 2. Strict JSON Instability

Large-block dVLM decoding can produce malformed JSON/tool calls.

Impact:

- strict parse fails
- repair may recover an action, but that is not strict model correctness

Required reporting:

- strict parse rate
- strict mobile_use rate
- repaired mobile_use rate
- repaired count/rate

### 3. Action-Type Collapse

Observed/possible collapse:

- overproduction of `swipe`
- underproduction of `click`
- missing `type` or `system_button`

Likely contributors:

- noisy branch undertraining
- class imbalance
- AITW dual-point click/swipe ambiguity
- repeated app/task loops

Required fix:

- action-balanced sampling or reweighting
- noisy branch KD
- structural/action token weighting
- per-action validation

### 4. Large-Block Coherence Failure

bd4/bd8/bd16/bd32 can fail even when bd1 or AR works.

Impact:

- tau sweep alone cannot solve the issue if the model never produces valid
  action structure
- training must include target block sizes and validation per block size

### 5. Benchmark Distribution Mismatch

Gmail/general overfit can help a demo but does not prove general AndroidWorld
capability.

Required fix:

- broader curated mobile SFT corpus
- standard AndroidWorld task evaluation
- leakage-aware reports

### 6. TPU Utilization Is Not Fully Solved

Low TPU utilization can come from:

- Python input stalls
- dynamic shapes
- small effective batch
- host preprocessing
- compile/fusion limitations

Do not claim TPU optimization is complete.

## What Not To Claim

Do not claim:

- AndroidWorld benchmark is complete.
- The model is paper-ready on AndroidWorld.
- Repaired output is strict output.
- Coordinate mismatch is the only failure.
- bd32-only training fixes all block sizes.
- Tau sweep fixes malformed large-block decoding.
- Steering module has been validated.
- 32B distillation is unnecessary forever.
- Gmail overfit proves broad mobile GUI competence.
- TPU v6e performance is saturated.

## Safe Current Claim

```text
The current work has identified and documented the main blockers for making
Fast-dVLM / GUI-Owl large-block decoding reliable: coordinate convention
alignment, noisy-branch KD, block-size-specific training, action balance,
strict JSON stability, and benchmark infrastructure. The current step-3000
checkpoint is a diagnostic artifact, not a final AndroidWorld result.
```
