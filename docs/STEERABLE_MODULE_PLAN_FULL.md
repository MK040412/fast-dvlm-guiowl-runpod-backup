# Steerable Module Design

Date: 2026-06-06

Target reader: Claude / future agents continuing the Fast-dVLM / GUI-Owl
project.

This document records the current design decision for the steerable module.

The main decision:

```text
The steerable module is NOT a denoise-trace-only adapter.

It is a KV-cache-aware residual controller for a frozen Fast-dVLM backbone.

It reads:
  1. persistent Fast-dVLM KV/cache state
  2. recent exact committed-token memory
  3. current block-local denoise trace
  4. current noisy block tokens
  5. timestep/block position

It outputs:
  small gated residuals injected into selected frozen Fast-dVLM layers.
```

The base GUI-Owl / Fast-dVLM model remains frozen.

---

## 1. One-Sentence Definition

The steerable module is a steering-only adapter placed on top of a frozen
bd32 Fast-dVLM / GUI-Owl backbone. It reads the frozen model's persistent KV
cache through a Kimi-style linear state, combines it with current block denoise
trace and recent exact memory, and injects small residuals into selected frozen
layers to stabilize `mobile_use` JSON, action type, and coordinate grounding.

---

## 2. Why We Need This Module

Fast-dVLM decoding is faster than pure AR decoding because it commits multiple
tokens per block. However, the current failure modes are:

```text
1. strict JSON/tool-call instability
2. malformed mobile_use output
3. action collapse, especially click/swipe imbalance
4. coordinate grounding instability
5. bd4/bd8/bd16/bd32 large-block incoherence
```

The base GUI-Owl / Fast-dVLM model already contains useful GUI action skills.
The steerable module should not learn a new mobile agent from scratch.

Instead:

```text
frozen Fast-dVLM = compressed skill library
steerable module = mode selector / residual controller
```

It should bias the frozen model toward the correct mode during diffusion:

```text
click vs swipe
valid JSON vs malformed JSON
coordinate field vs random number tokens
mobile_use schema vs free text
```

---

## 3. What Changed From the Earlier Design

Earlier design:

```text
denoise trace -> TraceEncoder -> Kimi steering -> residual
```

This is incomplete.

Reason:

- At the first denoise step, the denoise trace is empty.
- The real Fast-dVLM context lives in the persistent KV cache.
- Prompt, screenshot, instruction, prior actions, and future reasoning prefix
  are represented in the base KV/cache state.
- Ignoring KV cache makes steering blind to the actual long-range state of the
  frozen model.

Updated design:

```text
persistent KV/cache state
        +
recent exact committed memory
        +
current block denoise trace
        +
current noisy block tokens
        |
        v
Kimi/MLA steering fusion
        |
        v
gated zero-init residuals
        |
        v
frozen Fast-dVLM selected layers
```

---

## 4. Core Mental Model

```text
Frozen Fast-dVLM
  = already knows many GUI action skills.

Persistent KV state
  = long-term state: prompt, screenshot, instruction, previous actions,
    previous reasoning, committed blocks.

Recent exact memory
  = local syntax and recently committed JSON/action tokens.

Denoise trace
  = short-term evidence of what the current block is trying to become.

Steerable module
  = reads long-term state + local syntax + denoise evidence, then applies a
    small residual to guide the frozen model toward the right action mode.
```

---

## 5. Frozen / Trainable Boundary

### Frozen

The following must remain frozen:

```text
GUI-Owl / VLM encoder
Fast-dVLM backbone
token embeddings
all transformer layers
self-attention
cross-attention
FFN / MLP
LM head
tokenizer
original Fast-dVLM block-level cache/context path
base KV cache
```

Do not:

```text
unfreeze base weights
add LoRA to base
change tokenizer
replace the original Fast-dVLM cache path
modify base past_key_values in-place
train embeddings
train LM head
```

### Trainable

Trainable modules:

```text
PersistentKVStateReader / projector
RecentExactMemoryEncoder
TraceEncoder
KimiLinearState / MLA fusion blocks
zero-init residual projection heads
optional timestep gate
optional action/coordinate/structure auxiliary heads
```

Optimizer must see only steering parameters.

---

## 6. Whole-System Overview

```text
screenshot + instruction
        |
        v
+-------------------------------+
| FROZEN GUI-Owl / VLM encoder  |
+-------------------------------+
        |
        v
+--------------------------------------------------+
| FROZEN Fast-dVLM backbone                         |
| original block cache/context path remains intact  |
+--------------------------------------------------+
        |
        | persistent base KV cache
        v
+--------------------------------------------------+
| TRAINABLE Steerable Module                        |
|                                                  |
| Inputs:                                          |
|   - persistent KV linear state                   |
|   - recent exact memory                          |
|   - block-local denoise trace                    |
|   - current noisy block tokens                   |
|   - timestep / position                          |
|                                                  |
| Output:                                          |
|   - small gated residuals for selected layers    |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
| FROZEN Fast-dVLM layers with residual injection   |
+--------------------------------------------------+
        |
        v
mobile_use JSON/action/coordinate tokens
```

---

## 7. Block-by-Block Lifecycle

Fast-dVLM generates response/action tokens by blocks:

```text
[block 0] [block 1] [block 2] ...
```

For bd32:

```text
each block = 32 tokens
```

Each block is denoised:

```text
x_b,K -> x_b,K-1 -> ... -> x_b,1 -> x_b,0
noisy     less noisy              clean
```

Before block `b`:

```text
C_<b = original frozen base KV cache before block b
S_<b = steering persistent Kimi linear state before block b
R_<b = recent exact memory before block b
y_<b = previous committed tokens before block b
```

At block start:

```text
T_b   = empty TraceBank
x_b,K = [MASK] * bd
```

Denoise loop:

```text
for k = K ... 1:

    steering reads:
      S_<b       # persistent Kimi KV state
      R_<b       # recent exact memory
      T_b,>k     # previous denoise traces in this block
      x_b,k      # current noisy/masked block tokens
      k          # timestep

    steering computes:
      residuals r_b,k^l

    frozen base forward:
      logits_b,k, trace U_b,k = FastDVLMBasedForward(
          x_b,k,
          original_context=C_<b,
          residuals=r_b,k^l
      )

    append:
      T_b.append(detach(U_b,k))

    denoise update:
      x_b,k -> x_b,k-1
```

At block end:

```text
x_b,0 is committed
original base KV cache C_<b is updated as usual
steering persistent state S_<b is updated with new committed block KV
recent exact memory R_<b is updated
T_b is discarded
```

Next block:

```text
C_<b+1 = base KV cache including x_b,0
S_<b+1 = persistent Kimi state including x_b,0
R_<b+1 = recent exact memory including x_b,0
T_b+1 = empty
```

---

## 8. Persistent KV State

The persistent KV cache is central.

The original base model keeps its own `past_key_values` / KV cache for normal
decoding. The steerable module must not replace or alter this cache.

Instead:

```text
base KV cache
    |
    | detach read
    v
steering KV view
    |
    v
Kimi-style linear state
```

### Why Not Full Quadratic Attention?

If steering used full attention over all persistent cache tokens every denoise
step:

```text
cost ~= O(K * bd * L)
```

where:

```text
K  = denoise steps
bd = block size
L  = context length
```

If cache tokens were self-attended again, it could become even worse.

But the intended design is not quadratic.

The steering module uses a Kimi / linear-attention-style state:

```text
state = sum_i phi(K_i)^T V_i
norm  = sum_i phi(K_i)

read(Q):
  output = phi(Q) @ state / (phi(Q) @ norm)
```

So the persistent cache can be summarized incrementally:

```text
S_<b+1 = update(S_<b, KV(x_b,0))
```

Instead of rescanning all context from token 0 every block.

### Current Design Decision

Use:

```text
PersistentKVState S_<b
```

not:

```text
full raw KV cache as dense attention memory at every denoise step
```

This makes KV cache usage practical for the 2B model.

---

## 9. Recent Exact Memory

Linear KV state is efficient, but it may blur exact local syntax.

For tool-call generation, local syntax matters:

```text
{ } [ ] : ,
"name"
"mobile_use"
"arguments"
"action"
"coordinate"
```

Therefore keep a small exact memory:

```text
R_<b = recent committed tokens/KV
```

Recommended initial settings:

```text
recent_token_window = 128 or 256
source = previous committed response/action/reasoning tokens
```

Purpose:

```text
1. maintain JSON syntax continuity
2. preserve immediately preceding action context
3. preserve compact reasoning prefix if present
4. reduce structural drift in final JSON blocks
```

This memory can be encoded by:

```text
token embeddings
hidden states
selected recent KV
```

The base token embeddings remain frozen. If token embeddings are used, detach
them or use a separate trainable small embedding table only for steering if
needed.

---

## 10. Block-Local Denoise Trace

The denoise trace is still important, but it is not the only input.

TraceBank lifetime:

```text
Block b starts:
  T_b = empty

Step K:
  steering reads empty T_b
  base forward produces U_b,K
  T_b = {U_b,K}

Step K-1:
  steering reads {U_b,K}
  base forward produces U_b,K-1
  T_b = {U_b,K, U_b,K-1}

...

Block b ends:
  T_b is discarded
```

Trace contents can include compact summaries:

```text
hidden states
self-attention K/V summary
logits entropy
top-k logit summary
confidence
layer id
timestep
```

Do not store full-vocab logits.

Do not carry full denoise-history KV across blocks.

---

## 11. One-Step-Lag Rule

The trace produced at step `k` cannot be used to create the residual for the
same step `k`.

Correct:

```text
r_b,k uses {U_b,K, U_b,K-1, ..., U_b,k+1}
```

Incorrect:

```text
r_b,k uses U_b,k
```

Why:

The residual for a denoise step must be computed before that denoise step's
forward pass creates its trace.

---

## 12. Steering Fusion Architecture

Recommended modules:

```text
PersistentKVStateReader
RecentExactMemoryEncoder
TraceEncoder
CurrentBlockEncoder
KimiMLAFusion
ResidualHeads
```

### 12.1 PersistentKVStateReader

Input:

```text
S_<b: persistent Kimi linear state
```

Output:

```text
z_cache: [B, bd, steer_dim]
```

This is produced by reading the persistent state with queries derived from:

```text
current noisy block tokens
timestep embedding
block position embedding
trace summary
```

### 12.2 RecentExactMemoryEncoder

Input:

```text
R_<b: recent exact tokens/KV/hidden states
```

Output:

```text
z_recent: [B, bd, steer_dim]
```

This can be implemented as:

```text
small cross-attention over last N tokens
or pooled recent-token encoder
or Kimi linear state over recent window
```

### 12.3 TraceEncoder

Input:

```text
T_b,>k: [B, T_trace, bd, trace_dim]
```

Output:

```text
z_trace: [B, bd, steer_dim]
```

If `T_b` is empty:

```text
return learned empty_trace embedding expanded to [B, bd, steer_dim]
```

### 12.4 CurrentBlockEncoder

Input:

```text
x_b,k token ids
timestep k
position 0..bd-1
optional block index
```

Output:

```text
z_cur: [B, bd, steer_dim]
```

### 12.5 KimiMLAFusion

Combine:

```text
z_cache
z_recent
z_trace
z_cur
```

Recommended:

```text
z = KimiBlock(z_cur, memory=[z_cache, z_recent, z_trace])
z = SSDLinearBlock(z)
z = SSDLinearBlock(z)
z = SSDLinearBlock(z)
z = MLAAnchorBlock(z, anchors=[cache_anchor, trace_anchor, recent_anchor])
```

The exact implementation can be simplified for MVP, but must preserve the
inputs and frozen-base boundary.

---

## 13. Residual Injection

The steering module outputs residuals:

```text
r_b,k^l: [B, bd, hidden_dim]
```

for selected injection layers.

Recommended:

```text
inject_layers = last8
tap_layers = last4 or last8
```

Injection point:

Preferred:

```text
after MLP/FFN output before transformer block residual stream continues
```

Fallback:

```text
block output hook
```

If falling back, log:

```text
injection_point = "block_output_fallback"
```

Residual heads must be zero initialized:

```text
ZeroInitLinear_l: steer_dim -> hidden_dim
weight = 0
bias = 0
```

This ensures the initial steering-enabled model matches the base model.

---

## 14. Timestep Gate

Use a timestep gate so residuals are strongest during mode selection.

If denoise order is:

```text
K -> 1
```

then:

```text
progress = (K - k) / max(K - 1, 1)
gate(k) = exp(-0.5 * ((progress - gate_center) / gate_width)^2)
```

Defaults:

```text
gate_center = 0.35
gate_width = 0.15
learn_gate = false initially
```

Interpretation:

```text
early denoise:
  residual small because signal is very noisy

middle denoise:
  residual strongest because action/structure mode is selected

late denoise:
  residual smaller because many tokens are already committed
```

---

## 15. Parameter Size Target

The user expects roughly a 300M steering part as an upper target.

Do not force 300M in MVP if it creates instability.

Suggested stages:

### MVP

```text
steer_dim = 512 or 768
tap_layers = last4
inject_layers = last8
n_steer_blocks = 2
ssd_layers = 3
mla_layers = 1
recent_window = 128 or 256
```

Expected size:

```text
~80M-180M depending on hidden_dim and per-layer residual heads
```

### Strong Version

```text
steer_dim = 768 or 1024
tap_layers = last8
inject_layers = last8 or last12
n_steer_blocks = 3 or 4
recent_window = 256
more MLA anchors
```

Expected size:

```text
~200M-300M
```

Start smaller, prove identity/frozen checks, then scale.

---

## 16. Training Mode

Default training mode:

```text
teacher_forced_trace_sft
```

Procedure:

```text
1. Load image/instruction/target mobile_use JSON.
2. Tokenize target.
3. Split target into bd-sized blocks.
4. For selected block b, generate noisy/masked x_b,k using existing Fast-dVLM schedule.
5. Collect base trace with frozen base.
6. Build or update persistent KV state from frozen base cache.
7. Run frozen base with steering residuals.
8. Compute loss against clean target tokens.
9. Update steering params only.
```

Important:

Do not wrap the residual-consuming forward in `torch.no_grad()`.

Correct:

```python
base.requires_grad_(False)
logits = frozen_base(..., residuals=residuals)
loss = ce_loss(logits, target)
loss.backward()
optimizer.step()  # optimizer has steering params only
```

Incorrect:

```python
with torch.no_grad():
    logits = frozen_base(..., residuals=residuals)
```

This would break gradients to the steering module.

---

## 17. Loss

Main loss:

```text
L_ce = masked-token CE against clean target y_b
```

Additional losses:

```text
L_action = action-type classification CE
L_coord  = SmoothL1 over normalized coordinates if parseable
L_res    = residual norm penalty
```

Total:

```text
L =
  L_ce
+ lambda_action * L_action
+ lambda_coord  * L_coord
+ lambda_res    * L_res
```

Default:

```text
lambda_action = 0.1
lambda_coord  = 0.05
lambda_res    = 1e-4
```

Token weighting:

```text
structure/tool-call tokens: higher weight
coordinate field/number tokens: higher weight
action type tokens: higher weight
```

The goal is not fluent language. The goal is stable `mobile_use` structure,
correct action type, and grounded coordinates.

---

## 18. Reasoning/KD Compatibility

Reasoning can be added later as compact cache-prefix supervision.

Example:

```text
<plan>
subgoal: open Gmail inbox
target_element: Gmail app icon
action_type: click
coordinate_basis: center of Gmail icon
</plan>
<final>
{"name":"mobile_use","arguments":{"action":"click","coordinate":[521,843]}}
</final>
```

This is not ReACT.

ReACT:

```text
Thought -> Action -> Observation -> Thought -> ...
```

Our setting:

```text
compact reasoning prefix -> final mobile_use JSON
```

Reasoning tokens become committed tokens and enter the persistent KV state.
The steering module can then read them through:

```text
S_<b: persistent Kimi KV state
R_<b: recent exact memory
```

This is why KV-aware steering is more future-compatible than denoise-trace-only
steering.

If logit distillation is added later:

```text
32B-Think:
  compact plan / target element / coordinate rationale

32B-Instruct:
  final JSON/action token top-k logprobs
```

KD should focus on final JSON/action/coordinate tokens.

---

## 19. Sparse Attention Decision

Sparse attention is not the first required mechanism for 2B.

Reason:

The steering module should not run quadratic full attention over persistent KV.
It should use Kimi/linear state.

If implemented as:

```text
S_<b = linear state over persistent KV
read(S_<b, query)
```

then the cost is not `L^2`.

Potential bottleneck becomes:

```text
KV read bandwidth
number of tapped layers
denoise steps
batch size
state size
```

Therefore priority:

```text
1. Use Kimi linear persistent state.
2. Update state incrementally after each committed block.
3. Use recent exact memory for local syntax.
4. Add sparse/page selection only if linear state bandwidth is still too high.
```

If adding sparse later, apply it only to steering KV read path, not to base
Fast-dVLM attention.

Do not change base attention initially.

---

## 20. Safety Checks

Mandatory checks:

### Frozen Check

Verify:

```text
base_param_changed = false
base_grad_nonzero_count = 0
optimizer_params = steering params only
```

Save:

```text
frozen_check.json
```

### Zero-Init Identity Smoke

With residual heads zero-initialized:

```text
base logits == steering-enabled logits
```

Expected:

```text
max_abs_logit_diff <= small tolerance
mean_abs_logit_diff close to 0
```

If this fails, stop and fix before training.

### Trace Lifetime Check

Verify:

```text
TraceBank resets every block
full denoise trace is not carried across blocks
one-step-lag is enforced
```

### KV Boundary Check

Verify:

```text
base KV cache is read-only
steering uses detach view
base original cache path is unchanged
persistent steering state updates only from detached committed KV
```

---

## 21. Expected Benefits

This design should help because:

```text
1. It uses the actual long-range state of the frozen model.
2. It avoids blind first-step steering when denoise trace is empty.
3. It can leverage future reasoning prefix through KV/cache.
4. It keeps denoise trace for current block mode correction.
5. It preserves exact recent syntax for JSON/tool-call stability.
6. It avoids quadratic full-cache attention through Kimi linear state.
7. It leaves the base model and original Fast-dVLM cache path unchanged.
```

---

## 22. Main Risks

### Risk 1: Steering Becomes Too Heavy

If the persistent KV state reader is too large, steering overhead can erase
Fast-dVLM speed gains.

Mitigation:

```text
start with last4 tap layers
small steer_dim
block-static persistent state
incremental updates
recent window 128/256
measure latency
```

### Risk 2: Residual Destabilizes Base

Mitigation:

```text
zero-init residual heads
Gaussian timestep gate
residual norm penalty
identity smoke test
low initial LR
```

### Risk 3: Train/Decode Mismatch

Mitigation:

```text
same KV state construction in train and decode
same recent memory rule
same trace lifetime rule
same mRoPE/DeepStack path
```

### Risk 4: Reasoning Prefix Hurts Latency

Mitigation:

```text
compact structured plan only
20-60 token target
keep direct-action examples in mix
parser executes final JSON only
```

---

## 23. Recommended File Structure

If implementing:

```text
fast_dvlm/steering/
  __init__.py
  config.py
  persistent_kv_state.py
  recent_memory.py
  trace_bank.py
  trace_encoder.py
  kimi_linear.py
  kimi_trace_steering.py
  residual_injector.py
  wrapper.py
  metrics.py
  json_repair.py

tools/
  inspect_model_for_steering.py
  check_frozen.py
  smoke_steering_identity.py

train_steering_sft.py
eval_steering.py
```

---

## 24. Suggested Config

Initial config:

```json
{
  "bd": 32,
  "tap_layers": "last4",
  "inject_layers": "last8",
  "steer_dim": 512,
  "mla_dim": 256,
  "num_heads": 8,
  "ssd_layers": 3,
  "mla_layers": 1,
  "n_steer_blocks": 2,
  "persistent_kv": true,
  "persistent_kv_mode": "kimi_linear_state",
  "persistent_kv_update": "incremental_committed_blocks",
  "recent_exact_memory": true,
  "recent_token_window": 256,
  "denoise_trace": true,
  "trace_detach": true,
  "max_trace_steps": "all",
  "one_step_lag": true,
  "gate_type": "gaussian",
  "gate_center": 0.35,
  "gate_width": 0.15,
  "learn_gate": false,
  "zero_init_residual": true,
  "lambda_action": 0.1,
  "lambda_coord": 0.05,
  "lambda_res": 0.0001
}
```

Scale-up config:

```json
{
  "tap_layers": "last8",
  "inject_layers": "last8_or_last12",
  "steer_dim": 768,
  "mla_dim": 384,
  "n_steer_blocks": 3,
  "recent_token_window": 256
}
```

---

## 25. Acceptance Criteria

The steerable module is acceptable only if:

```text
1. Base checkpoint files are not modified.
2. All base parameters remain frozen.
3. Optimizer sees steering params only.
4. Zero-init identity smoke test passes.
5. Persistent KV state is detach/read-only relative to base.
6. Original Fast-dVLM cache path remains unchanged.
7. TraceBank resets per block.
8. One-step-lag rule is enforced.
9. Adapter-only checkpoint is saved.
10. Eval reports strict and repaired metrics separately.
11. Latency overhead is measured.
12. Logs include trace_source, kv_state_mode, injection_point, tap_layers,
    inject_layers, trainable parameter count, residual norms, and gate stats.
```

---

## 26. Final Design Summary

Final selected design:

```text
KV-cache-aware Kimi-linear steerable residual controller.
```

It is not:

```text
denoise-trace-only adapter
full-cache quadratic attention module
LoRA on the base model
replacement of Fast-dVLM cache path
new action generator from scratch
```

It is:

```text
1. persistent KV state reader
2. recent exact memory encoder
3. block-local denoise trace encoder
4. Kimi/MLA fusion module
5. zero-init gated residual injector
```

The most important idea:

```text
Fast-dVLM's KV cache is the real long-term state.
Denoise trace is the current block's local uncertainty.
Steering must read both.
```

This design preserves the frozen base model while allowing a trainable
controller to improve strict tool-call structure, action type selection, and
coordinate grounding.
