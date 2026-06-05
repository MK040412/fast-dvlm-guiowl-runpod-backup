# Data Curation Plan

This document records the current data-curation plan for mobile GUI SFT.

The current task is SFT data curation and distillation preparation, not RL.

## Goal

Build a high-quality mobile GUI SFT corpus that is compatible with GUI-Owl-1.5
and useful for Fast-dVLM block-diffusion training.

The corpus should improve:

- strict JSON/tool-call validity
- valid `mobile_use` schema
- action-type coverage
- coordinate grounding
- block-diffusion large-block decoding stability
- future reasoning/KD/steering compatibility

## Canonical Target Schema

All final targets must use:

```text
coord_mode: guiowl_norm1000_xy
order: [x, y]
range: 0..1000
schema: mobile_use
```

Pixel labels must be converted before SFT.

## Existing AiTW Status

AiTW / `jacklishufan/aitw` should not be blindly reprocessed as the main work
item because user-owned processed/classified assets already exist.

Important existing artifact:

```text
HF dataset: KMK040412/aitw-androidworld-overfit-mix
approx rows: 476,043
coordinate mode: norm1000
parse failures: 0 in the reported processed mix
out-of-range coordinates: 0 in the reported processed mix
```

Observed action distribution in the processed mix:

```text
click          about 250k
swipe          about 95k
terminate      about 55k
system_button  about 37k
type           about 37k
```

This mix is useful for:

- AndroidWorld-core overfit/smoke
- norm1000 validation reference
- action-balanced SFT seed data

It should not be treated as a broad AndroidWorld generalization benchmark.

## Why More Raw Rows Are Not Enough

The observed failures are not only lack of data:

- coordinate convention mismatch
- malformed JSON
- action collapse
- repeated goals dominating the gradient
- noisy branch undertraining
- block-level coherence failures

Therefore curation should optimize for quality and coverage, not only row count.

## Curation Rules

Every source should be canonicalized through the same steps:

1. Parse raw action.
2. Map to `mobile_use`.
3. Detect coordinate convention.
4. Convert to norm1000 `[x, y]`.
5. Validate coordinate range.
6. Validate JSON schema.
7. Preserve trajectory history if available.
8. Add source metadata.
9. Deduplicate near-identical goals and repeated screens.
10. Balance action types.
11. Cap dominant app/task clusters.
12. Produce a per-source validation report.

Required row fields:

```text
id
source_dataset
source_split
app_or_domain
task_family
instruction
screenshot or image reference
history
target_json
action_type
coord_mode
screen_width
screen_height
quality_score
leakage_tag
```

## Action Balance

The corpus must avoid collapse to one action type.

Track:

- click
- swipe
- type
- system_button/back/home
- terminate/done
- observe/wait, if used

AITW-style `DUAL_POINT` actions must be split into click vs swipe by comparing
touch and lift points. Do not keep ambiguous action labels if they hide
click/swipe imbalance.

## Dataset Direction

Candidate public mobile GUI data families:

- AndroidControl-like mobile action trajectories
- AMEX-like mobile grounding/action chains
- GUI-Odyssey-like long-horizon multi-app trajectories
- OpenMobile-style mobile task data
- static GUI grounding datasets as auxiliary grounding data

Static grounding datasets should help coordinate grounding but should not be
reported as dynamic AndroidWorld success.

## Distillation Direction

Current teacher:

```text
GUI-Owl-1.5-2B AR / same-model clean branch
```

Future teachers:

```text
mPLUG/GUI-Owl-1.5-32B-Instruct
mPLUG/GUI-Owl-1.5-32B-Think
```

Reasoning/Think distillation should be prepared as a future recipe:

- preserve reasoning fields if generated
- keep action target separate from rationale
- do not force a ReAct-only schema if the deployed model benefits from KV-cache
  accumulated reasoning/history
- logit distillation can be added after basic action validity is stable

## Time-Budgeted Mixes

Recommended deliverables:

```text
5h mix:
  small, high-quality, action-balanced mobile SFT mix
  goal: fast sanity and overfit capability

10h mix:
  add more apps/task families and type/system_button coverage
  goal: reduce action collapse

20h mix:
  add long-horizon and multi-app trajectories
  goal: improve AndroidWorld-style task progress

40h mix:
  broad mobile GUI corpus with reasoning/KD-ready columns
  goal: paper-grade training input
```

Every mix should have:

- parquet shards
- dataset card
- examples visible in Hugging Face Dataset Viewer
- action distribution report
- coordinate validation report
- source caps/dedup report
- leakage/decontamination notes

## Non-Claims

Do not claim:

- curation alone proves AndroidWorld success
- Gmail overfit generalizes to all mobile tasks
- static grounding accuracy equals dynamic task success
- 32B teacher is already integrated
- reasoning distillation is already validated

Safe claim:

```text
The curation target is a GUI-Owl-compatible mobile SFT corpus with validated
norm1000 coordinates, balanced action types, explicit source metadata, and
future-ready columns for reasoning and KD.
```
