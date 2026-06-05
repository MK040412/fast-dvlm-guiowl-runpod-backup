# Data Curation Plan for Fast-dVLM / GUI-Owl Mobile SFT

Date: 2026-06-06

Target reader: Claude / future agents continuing this project.

This document records the current decisions, reasons, failure analysis, dataset
selection, curation rules, and expected deliverables for building a mobile GUI
SFT dataset for Fast-dVLM / GUI-Owl.

The immediate task is **SFT data curation**. Do not treat this as an RL rollout
task or an architecture rewrite task.

---

## 1. Project Goal

We are training a Fast-dVLM / BARD-style block-diffusion version of
GUI-Owl-1.5-2B for mobile GUI control.

The model should emit strict GUI-Owl style `mobile_use` tool calls:

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "click",
    "coordinate": [521, 843]
  }
}
```

Swipe example:

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

The SFT dataset should improve:

1. strict JSON/tool-call validity
2. correct `mobile_use` schema
3. action-type coverage: click, swipe, type, system_button, terminate
4. coordinate grounding
5. block-diffusion large-block decoding stability at bd4/bd8/bd16/bd32
6. compatibility with future reasoning/KD distillation

---

## 2. Absolute Coordinate Contract

All final SFT targets must use the GUI-Owl-1.5 coordinate convention:

```text
coord_mode: guiowl_norm1000_xy
coordinate order: [x, y]
range: 0..1000
origin: top-left
```

Pixel coordinates are forbidden in final targets unless explicitly converted.

Only the executor maps normalized coordinates to device pixels:

```text
x_pixel = round(x_norm1000 / 1000 * screen_width)
y_pixel = round(y_norm1000 / 1000 * screen_height)
```

If source data is pixel-based:

```text
x_norm1000 = round(x_pixel / screen_width  * 1000)
y_norm1000 = round(y_pixel / screen_height * 1000)
```

After conversion:

```text
0 <= x <= 1000
0 <= y <= 1000
```

For swipe, validate both `coordinate` and `coordinate2`.

Do not silently accept `[y, x]`. Coordinate order must be `[x, y]`.

---

## 3. Historical Failure That Must Not Repeat

A previous Gmail SFT run accidentally trained on pixel coordinates for Gmail
while mixing a small amount of norm1000 general data.

Observed situation:

- Gmail data target JSON contained pixel coordinates.
- General rows used norm1000.
- GUI-Owl teacher / AndroidWorld executor expect norm1000.
- Result: model emitted coordinates compressed toward the upper-left region
  under AndroidWorld interpretation.

This caused apparent spatial grounding failures even when action format looked
partially correct.

Therefore every curated shard must include:

- coordinate mode detection
- conversion report
- out-of-range report
- examples before/after conversion

No dataset should be used in SFT without passing this validation.

---

## 4. Current Data Status

AiTW / `jacklishufan/aitw` should not be reprocessed as the main task here.

Reason:

- The user already processed/labeled/classified AiTW on Hugging Face.
- We already have an AiTW/AndroidWorld overfit mix with norm1000 coordinates.
- Spending Vultr time on raw AiTW again is redundant.

Existing processed dataset:

```text
HF dataset: KMK040412/aitw-androidworld-overfit-mix
approx rows: 476,043
coord_mode: norm1000
parse failures: 0
out-of-range coords: 0
```

Previously observed action distribution:

```text
click         ~250,740
swipe          ~95,585
terminate      ~55,349
system_button  ~37,336
type           ~37,033
```

This existing dataset can be used as:

- seed data
- reference schema
- optional mix component

But the current curation job should focus on other public mobile-device datasets
that approximate the Mobile-Agent-v3.5 / GUI-Owl-1.5 mobile data flywheel.

---

## 5. Main Curation Decision

Do not blindly add more raw data.

The target is a clean, GUI-Owl-compatible, mobile-device SFT corpus.

The data should be curated to reduce the exact failure modes we observed:

1. coordinate convention mismatch
2. malformed JSON/tool-call output
3. action collapse, especially collapse to swipe
4. weak click/type/system_button coverage
5. poor large-block dVLM decoding coherence
6. app/task overfitting, especially repeated Gmail/settings loops

The curation strategy should follow a Mobile-Agent-v3.5-style public data
flywheel:

```text
public mobile GUI datasets
  -> canonical GUI-Owl mobile_use schema
  -> coordinate validation
  -> action mapping
  -> dedup / app-task balance
  -> quality scoring
  -> time-budgeted SFT mixes
  -> HF upload with reports
```

---

## 6. Relevant Literature and Rationale

This project should be guided by the following papers and design lessons.

### Mobile-Agent-v3.5 / GUI-Owl-1.5

Key lesson:

- GUI-Owl is trained around a data flywheel:
  - simulated environments
  - cloud sandbox environments
  - task/trajectory generation
  - unified thought synthesis
  - scalable environment RL
  - multi-platform GUI control

The exact internal training data is not fully public. Therefore, the best
public approximation is to curate mobile-device datasets with strict
schema/coordinate/action validation.

### UI-TARS / UI-TARS-2

Key lesson:

- GUI agents need perception, unified action modeling, and reasoning.
- The data flywheel and rollout/evaluation infrastructure are central.
- However, for the current step, direct action SFT should be stabilized first.

### SeeClick / ScreenSpot-Pro / OS-ATLAS / Aguvis

Key lesson:

- GUI grounding is a first-order bottleneck.
- Small-target and high-resolution GUI grounding are hard.
- Coordinate/action data quality matters more than raw row count.

### AndroidWorld / MobileWorld

Key lesson:

- Dynamic benchmark success is different from static SFT loss.
- AndroidWorld should guide task taxonomy, verifier design, and leakage checks.
- Do not directly leak standard benchmark tasks into train.

### AMEX / GUI-Odyssey / AndroidControl / OpenMobile

Key lesson:

- Public mobile data should be combined by role:
  - AndroidControl: direct mobile control
  - AMEX: grounding and action-chain annotations
  - GUI-Odyssey: cross-app and long-horizon navigation
  - OpenMobile: public mobile data synthesis/flywheel approximation

### Tulu 3 / Fixing It in Post / Influence Distillation / LLM2LLM / ACED

Key lesson:

- Curated high-quality data can beat larger noisy data.
- Metadata, quality tags, decontamination, and transparent reports are essential.
- Student-failure-driven data selection is useful.

### DeepSeek-R1 / s1 / LIMO

Key lesson for future work:

- Reasoning distillation can help, but not as long raw free-form CoT by default.
- Use compact, validated reasoning templates if adding reasoning later.

---

## 7. Datasets to Consider Now

Since AiTW is already handled, focus on the following public mobile-device
datasets.

### 7.1 AndroidControl

Priority: highest.

Why:

- Directly about mobile GUI control.
- Better instruction/action alignment than many raw web datasets.
- Useful for AndroidWorld-like mobile behavior.

Use for:

- direct action SFT
- high-level / low-level instruction alignment
- click/swipe/type/system_button coverage
- app/task diversity

Expected processing:

- map actions to `mobile_use`
- normalize coordinates to norm1000
- preserve high-level and low-level instruction metadata
- tag source reliability

### 7.2 AMEX

Priority: high.

Why:

- Mobile GUI multi-annotation dataset.
- Strong for grounding, functionality, and action-chain supervision.

Use for:

- target element grounding
- action-chain examples
- coordinate/action consistency
- future reasoning fields such as `target_element` and `coordinate_basis`

Expected processing:

- convert element/action annotations into mobile_use-compatible targets
- keep metadata for element/function descriptions
- use for grounding-heavy subset

### 7.3 GUI-Odyssey

Priority: medium-high.

Why:

- Cross-app mobile navigation.
- Useful against Gmail/general overfitting.
- Helps history-conditioned action and long-horizon behavior.

Use for:

- multi-step mobile tasks
- cross-app navigation
- previous-action/history conditioning
- KV/cache conditioning in causal/Fast-dVLM models

Expected processing:

- keep episode order and history fields
- preserve app/task tags
- dedup repeated navigation loops

### 7.4 OpenMobile

Priority: high if accessible.

Why:

- Public attempt to reproduce closed mobile-agent data recipes.
- Uses grounded instruction synthesis and trajectory generation.
- Most similar in spirit to Mobile-Agent-v3.5's data flywheel.

Use for:

- synthetic but grounded mobile tasks
- broader app functionality coverage
- recovery/error-correction trajectories
- task diversity

Expected processing:

- validate generated trajectories
- keep only rows with clear mobile screenshots/actions
- tag synthetic vs real
- downweight if noisy

### 7.5 AndroidWorld / MobileWorld

Priority: eval taxonomy, leakage control, verifier reference.

Do not treat standard eval tasks as ordinary train data.

Use for:

- app/task taxonomy
- leakage checks
- verifier-aligned task categories
- AndroidWorld-like but non-overlapping training examples

### 7.6 ScreenSpot / RICO / MobileViews-like grounding datasets

Priority: optional.

Use only if:

- screenshot is mobile
- coordinate can be converted to norm1000
- label can become grounding/action supervision

Do not let grounding-only data dominate direct action SFT.

---

## 8. Target HF Dataset Structure

Recommended HF dataset:

```text
KMK040412/mobile-gui-sft-mobileagent35-style-v1
```

Directory structure:

```text
canonical_full/
  shard-0000.parquet
  shard-0001.parquet
  ...

mixes/
  mix_5h/
    shard-0000.parquet
    ...
  mix_10h/
    ...
  mix_20h/
    ...
  mix_40h/
    ...

source_reports/
  coordinate_report.json
  action_distribution.json
  app_distribution.json
  task_distribution.json
  dedup_report.json
  drop_reason_report.json
  source_coverage_report.json
  leakage_report.json
  mix_sampling_report.json

recipes/
  curation_recipe.md
  schema.md
  coordinate_convention.md
  training_mix_recommendations.md

README.md
```

The HF dataset card should be in English and should include example rows that
are visible in the Hugging Face Dataset Viewer.

---

## 9. Canonical Schema

Every row should be converted to this schema or a superset of it:

```text
source: string
source_dataset_version: string
episode_id: string
step_id: string or int

app: string
task: string
instruction: string
history: json string or list

screenshot: image field / bytes / path
screen_width: int
screen_height: int

target_json: string
action_type: string
coordinate: list[int] or null
coordinate2: list[int] or null
coord_mode: "guiowl_norm1000_xy"

is_mobile_device: bool
is_eval_leakage_risk: bool
quality_score: float
quality_tags: list[string]
drop_reason: null or string

reasoning_plan: null
target_element: null
coordinate_basis: null
teacher_think_model: null
teacher_instruct_model: null
teacher_logprob_cache: null
```

`target_json` must be a strict JSON string parseable by Python `json.loads`.

The final target must be executable as `mobile_use`. Repair outputs should not
be stored as the SFT target.

---

## 10. Action Mapping

Supported `action_type` values:

```text
click
swipe
type
system_button
terminate
wait
unknown
```

### Click

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "click",
    "coordinate": [x, y]
  }
}
```

### Swipe

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "swipe",
    "coordinate": [x1, y1],
    "coordinate2": [x2, y2]
  }
}
```

### Type

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "type",
    "text": "..."
  }
}
```

### System Button

Map source semantics to:

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "back"
  }
}
```

or:

```text
home
enter
recents
```

depending on source semantics.

### Terminate

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "done"
  }
}
```

Do not silently guess ambiguous mappings. If uncertain:

- tag `ambiguous_action`
- lower quality score
- drop if action cannot be safely mapped

---

## 11. Action Balance Target

Earlier dVLM failure showed collapse toward `swipe`, with poor strict JSON and
no reliable click emission at useful block sizes.

Therefore action balance is required, not cosmetic.

Approximate target distribution for train mixes:

```text
click          45-55%
swipe          15-25%
type           10-15%
system_button   8-12%
terminate       5-10%
wait/other       small
```

This is not a strict law. If a mix deviates, write down why in
`mix_sampling_report.json`.

---

## 12. Quality Filtering

Hard drop:

```text
missing screenshot
non-mobile screenshot unless explicitly intended
invalid target action
invalid coordinate
coordinate convention impossible to determine
malformed final JSON target
duplicate over cap
high eval leakage risk
action cannot be mapped to mobile_use
empty or meaningless instruction
```

Soft tags:

```text
repeated_instruction
ambiguous_click_swipe
low_source_confidence
ocr_heavy
tiny_target
long_horizon
cross_app
recovery_trajectory
type_heavy
system_button_heavy
app_open_bootstrap
synthetic
teacher_verified
student_hard_case
```

---

## 13. Deduplication and Caps

Use deduplication to prevent repeated app/task loops from dominating.

Recommended dedup keys:

```text
source
app
normalized_instruction
action_type
rounded_coordinate_bucket
screenshot perceptual hash if available
trajectory action-sequence hash if available
```

Apply caps:

```text
max rows per exact instruction
max rows per app/task/action template
max rows per screenshot hash
max rows per repeated source goal
```

Avoid the previous failure mode where Gmail settings/toggle tasks repeated many
times and narrowed the model.

---

## 14. Quality Score

Assign a `quality_score` per row:

```text
score =
  0.20 * schema_valid
+ 0.20 * coordinate_valid
+ 0.15 * action_balance_value
+ 0.15 * app_task_diversity
+ 0.10 * grounding_difficulty
+ 0.10 * source_reliability
+ 0.05 * strict_json_simplicity
+ 0.05 * student_hard_case_tag
```

Definitions:

- `schema_valid`: strict JSON + mobile_use action parse passes.
- `coordinate_valid`: norm1000 coordinate validation passes.
- `action_balance_value`: higher for underrepresented valid actions.
- `app_task_diversity`: lower for repeated app/task templates.
- `grounding_difficulty`: higher for valid small-target/icon/text-field clicks.
- `source_reliability`: based on dataset reliability and parse certainty.
- `strict_json_simplicity`: output is simple and canonical.
- `student_hard_case_tag`: optional; higher if current dVLM fails but teacher/source succeeds.

Do not force low-quality rows into a mix just to hit row count.

---

## 15. Time-Budgeted Mixes

Build several mixes so training can choose based on TPU budget.

Since AiTW is already handled, these mixes mainly use:

- AndroidControl
- AMEX
- GUI-Odyssey
- OpenMobile
- optional mobile grounding datasets
- optional current processed seed dataset if needed

### 5h Mix

Target: 280k-380k rows.

Suggested composition:

```text
AndroidControl              100k-130k
AMEX                         60k-80k
GUI-Odyssey                  40k-60k
OpenMobile / synthetic       40k-60k
grounding-only converted     20k-30k
tool/open-app/type/system    20k
```

Purpose:

- maximize clean action supervision
- fix strict JSON and action collapse quickly
- avoid app/task overfitting

### 10h Mix

Target: 600k-800k rows.

Suggested composition:

```text
AndroidControl              200k+
AMEX                        120k+
GUI-Odyssey                 100k+
OpenMobile                  120k+
grounding/action balance     50k+
```

Purpose:

- stronger action diversity
- broader mobile app coverage
- more long-horizon/cross-app examples

### 20h Mix

Target: 1.2M-1.6M rows.

Purpose:

- stronger generalization
- larger app/task coverage
- more type/system_button examples
- more valid long-horizon trajectories

### 40h Mix

Target: 2.5M-3.5M rows if enough clean data exists.

Purpose:

- research-grade broad SFT base
- multi-source mobile GUI corpus
- keep strict quality gates

Do not include low-quality rows just to reach 40h scale.

---

## 16. Reasoning and Logit Distillation Reserved for Later

Reasoning is important, but it is not the immediate curation target.

The current SFT dataset should reserve columns for future reasoning/logit
distillation:

```text
reasoning_plan = null
target_element = null
coordinate_basis = null
teacher_think_model = null
teacher_instruct_model = null
teacher_logprob_cache = null
```

Future plan:

- Serve `mPLUG/GUI-Owl-1.5-32B-Think` with vLLM.
- Use it to generate compact structured reasoning, not long free-form CoT.
- Serve `mPLUG/GUI-Owl-1.5-32B-Instruct` for final action token logprobs.
- Cache top-k logprobs offline.
- Do not query 32B teacher online during TPU training.

Recommended future target format:

```text
<plan>
subgoal: ...
target_element: ...
action_type: ...
coordinate_basis: ...
</plan>
<final>
{"name":"mobile_use","arguments":{...}}
</final>
```

The executable output remains only the JSON inside `<final>`.

Do not train raw long ReACT traces by default. Our model uses KV/cache context;
reasoning should be a compact cache-prefix supervision that stabilizes final
tool-call generation.

---

## 17. Logit Distillation Reserved Fields

If adding teacher logprobs later, store compact top-k cache:

```text
topk_token_ids_final
topk_logprobs_final
gold_token_id
gold_token_logprob
teacher_model
teacher_temperature
teacher_prompt_hash
```

Do not store full-vocab logits.

Recommended:

```text
topk = 16 or 32
always include gold token
KD mainly on final JSON/action/coordinate tokens
optional low-weight KD on plan tokens
```

For noisy dVLM branch, gate KD to medium/late denoise steps because AR teacher
distribution may be too aspirational at very high mask rates.

---

## 18. Required Reports

Every HF upload must include reports.

### coordinate_report.json

Include:

```text
coord modes detected
pixel->norm1000 conversion count
already norm1000 count
invalid coord count
out-of-range count
x/y min/max/mean/percentiles
per-source coordinate stats
examples before/after conversion
```

### action_distribution.json

Include:

```text
global action distribution
per-source action distribution
per-app action distribution
click/swipe/type/system_button/terminate counts
```

### app_distribution.json

Include:

```text
app counts
dominant apps
capped apps
app entropy or diversity summary
```

### task_distribution.json

Include:

```text
instruction template counts
duplicate task counts
long-horizon task counts
cross-app task counts
```

### dedup_report.json

Include:

```text
before/after row counts
duplicate keys used
rows removed per dedup reason
cap settings
```

### drop_reason_report.json

Include:

```text
invalid coordinate
invalid JSON
missing screenshot
unmappable action
eval leakage risk
duplicate cap
source parse failure
```

### leakage_report.json

Include:

```text
AndroidWorld/MobileWorld overlap heuristic
exact task match count
fuzzy instruction match count
app/task overlap notes
whether standard eval tasks were excluded
```

### mix_sampling_report.json

Include:

```text
source composition
action composition
app composition
quality score percentiles
row counts
sampling weights
reason for deviations from target distribution
```

---

## 19. Implementation Guidance for Vultr

Use Vultr as a curation worker.

Recommended environment:

```bash
uv venv
source .venv/bin/activate
uv pip install polars pyarrow datasets huggingface_hub pillow tqdm rich
uv pip install imagehash  # optional
```

Use `polars` and `pyarrow`, not pandas, for large parquet processing.

Process by shards:

1. download/source ingest
2. per-source canonicalization
3. per-shard coordinate validation
4. per-shard action mapping
5. per-shard quality scoring
6. merge metadata
7. dedup/cap
8. sample mixes
9. write parquet shards
10. upload to HF

Use CPU/RAM efficiently:

- parallel per-source processing
- parallel per-shard processing
- avoid loading all images into memory
- write intermediate parquet files
- upload periodically

Do not include secrets in logs or dataset cards.

---

## 20. Recommended HF Upload Deliverables

At the end, report:

```text
HF dataset URL
commit id
canonical_full row count
mix_5h row count
mix_10h row count
mix_20h row count
mix_40h row count if created
source composition
action distribution
coordinate validation summary
dedup summary
leakage summary
files uploaded
recommended TPU training command / mix choice
```

---

## 21. TPU Training Implications

The curated dataset should feed the existing TPU Fast-dVLM/KD training recipe.

Recent TPU recipe context:

```text
bd schedule: 4,8,16,32 mixed
dtype: bf16
loss includes noisy CE, clean CE, noisy KD
mRoPE and DeepStack exact path should remain enabled
coordinate targets must be norm1000
```

Data quality is more important than raw loss alone.

Evaluate separately:

```text
strict JSON rate
mobile_use valid rate
repaired valid rate
action-type distribution
ground@100
coord L2
task success
latency
bd-specific behavior
```

Do not merge strict and repaired metrics. Repaired output is not strict model
correctness.

---

## 22. Final Decision Summary

Current immediate action:

```text
Build a clean direct-action mobile GUI SFT dataset.
Do not reprocess raw AiTW as main work.
Use AndroidControl + AMEX + GUI-Odyssey + OpenMobile where possible.
Canonicalize everything to GUI-Owl norm1000 mobile_use JSON.
Balance actions and apps.
Dedup repeated goals.
Upload HF dataset with detailed reports.
Reserve columns for later reasoning/KD.
```

Main reason:

The model's biggest practical failures are not solved by raw row count:

- coordinate convention mismatch
- strict JSON weakness
- action collapse
- app/task overfitting
- large-block decoding incoherence

The curation must directly target those failures.

If later adding reasoning:

```text
Use 32B-Think for compact structured action reasoning.
Use 32B-Instruct for final action token logprob KD.
Keep final executable target strict mobile_use JSON.
Cache teacher outputs offline.
```

This keeps the current dataset useful both for direct SFT now and reasoning/KD
distillation later.

---

## 18. SFT-vs-32B-Logit Dataset Strategy (added by Claude, 2026-06-06)

One curated corpus, TWO consumers:

- **SFT (current self-distillation way, unchanged)** consumes the **FULL** canonical
  corpus. Loss stays `1.0*CE_noisy + 0.75*CE_clean + 0.25*KD_noisy` (clean AR branch
  teaches noisy diffusion branch; intra-model). Hard-label `mobile_use` target.
- **32B logit distillation** consumes only a **high-value SUBSET** (32B inference is
  expensive, cannot run on millions of rows). Top-k logits are cached OFFLINE and the
  student adds a KL term ONLY on rows that have a cache.

Combined per-row loss at training time:
- row WITH cache  -> `self-distill SFT + KL(student_action_logits || 32B top-k logits)`
- row WITHOUT cache -> `self-distill SFT only`
(32B and 2B share the Qwen tokenizer/vocab -> cross-model logit KD is vocab-compatible.)

### Which dataset feeds what

| dataset | SFT weight | 32B-logit priority | why |
| --- | --- | --- | --- |
| AndroidControl | highest (backbone) | HIGH | hi/lo instruction alignment -> needs planning; 32B distribution valuable |
| GUI-Odyssey | medium | HIGHEST | cross-app / long-horizon = where 32B reasoning >> 2B; history-conditioned |
| AMEX | med-high | MEDIUM | small-target grounding + action-chains (soft dist helps hard grounding) |
| OpenMobile | medium | MEDIUM (recovery) | error-recovery trajectories = ambiguous decisions -> soft dist useful |
| grounding (ScreenSpot/RICO/MobileViews) | support | NONE | pure coordinate -> hard label suffices; do not waste 32B |
| aw-androidworld-overfit-mix (existing) | target sharpening | HIGH (4 AW-core) | inject 32B quality into demo/bench target tasks |

### Selecting the 32B-logit subset (tag during curation)
- Prefer **hard-case + diversity**: small targets, multi-step, cross-app, recovery, type-heavy.
- Reuse schema fields `student_hard_case_tag` / `quality_score` / `quality_tags` to set a
  new boolean flag `logit_distill_candidate`.
- Per-source caps so GUI-Odyssey + AndroidControl-highlevel dominate the subset; exclude
  easy/abundant rows.
- Target subset size: **~50k-150k rows** (controls 32B inference cost).

---

## 19. Offline 32B Logit Generation on TPU via vLLM (added by Claude, 2026-06-06)

Plan: generate 32B reasoning-model logits **batch-wise, OFFLINE**, cache top-k, then
distill. The 32B is NOT served online during 2B TPU training (read cache only).

NEW option: **vLLM TPU (`vllm-project/tpu-inference`, unified JAX/PyTorch path)** can run
Qwen3-VL-class models on TPU, so the 32B logit-cache pass can run on a TPU instead of a
separate GPU box (resource reuse).

CAVEAT (verified 2026-06): in the vLLM-TPU support matrix **Qwen3-VL is newly added and
listed as under-validation** (e.g. Qwen3-VL-8B-Instruct flagged uncertain), while
Qwen2.5-VL has full TPU support. GUI-Owl-1.5-32B is Qwen3-VL-based -> **validate a small
32B logit run on vLLM-TPU before committing**; fall back to a GPU box (RunPod H100) if
Qwen3-VL-32B is not yet stable on vLLM-TPU.

Workflow:
1. Curate corpus on Vultr CPU; tag `logit_distill_candidate` subset.
2. Serve `mPLUG/GUI-Owl-1.5-32B-Instruct` (and optionally `-32B-Think` for compact
   `<plan>` prefixes) via vLLM (TPU if validated, else GPU box), batched.
3. For each subset row, capture **top-k (~20) logprobs per action/plan token** (NOT
   full-vocab) into `teacher_logprob_cache`; store plan text in `reasoning_plan`.
4. 2B dVLM SFT on TPU reads the cache and adds the KL term (temp ~1-2) on action tokens.

Sources: vLLM TPU (`vllm-project/tpu-inference`), vLLM TPU recommended-models docs
(Qwen3-VL support status).
