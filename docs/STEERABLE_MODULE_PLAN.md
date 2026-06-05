# Steerable Module Plan

This is the current design plan for the future steering module. It is not a
claim that the steering module has already been validated.

## One-Sentence Definition

The steerable module is a steering-only adapter placed on top of a frozen bd32
Fast-dVLM / GUI-Owl backbone. It reads compressed persistent KV/cache state,
recent committed tokens, block-local denoise trace, current noisy tokens, and
timestep information, then injects small gated residuals into selected frozen
layers.

## Core Mental Model

```text
frozen Fast-dVLM
  = compressed GUI action skill library

persistent KV/cache state
  = long-range prompt/screenshot/instruction/action history state

recent exact committed tokens
  = local syntax and immediate JSON/action context

block-local denoise trace
  = evidence of what the current block is trying to become

steerable module
  = residual mode selector/controller
```

The module should not learn a mobile agent from scratch. It should steer modes
already supported by the frozen base:

- valid JSON vs malformed text
- `mobile_use` vs free-form assistant text
- click vs swipe
- coordinate field vs random number tokens
- terminate/type/system_button where appropriate

## Why Denoise Trace Alone Is Not Enough

Earlier design:

```text
denoise trace -> TraceEncoder -> Kimi steering -> residual
```

This is incomplete because:

- at the first denoise step the trace is empty
- the real task state lives in the base prompt/context/KV path
- previous committed blocks and action history matter
- JSON syntax depends heavily on recent committed tokens

Updated design:

```text
persistent KV/cache summary
      +
recent exact token memory
      +
block-local denoise trace
      +
current noisy tokens
      +
timestep / position
      |
      v
Kimi/MLA steering fusion
      |
      v
zero-init gated residual heads
      |
      v
frozen Fast-dVLM layers
```

## Frozen Boundary

Frozen:

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
original Fast-dVLM cache/context path
base KV cache tensors
```

Do not:

- unfreeze base weights
- train embeddings
- train LM head
- add LoRA to the base in this steering version
- change tokenizer
- replace original Fast-dVLM block cache/context
- carry full denoise-history KV across blocks

Trainable:

```text
PersistentKVStateReader / projector
RecentExactMemoryEncoder
TraceEncoder
Kimi linear state / SSD blocks
MLA fusion blocks
zero-init residual heads
timestep gates
optional action/coord/structure heads
```

## Block Lifecycle

For each block `b`:

```text
Before block:
  C_<b = frozen base KV/cache state
  S_<b = compressed steering state
  R_<b = recent exact token memory
  T_b  = empty trace bank

Denoise step k:
  steering reads S_<b, R_<b, T_b, x_b,k, k
  steering outputs residuals r_b,k
  frozen base forward consumes x_b,k + original context + residuals
  base produces logits and trace U_b,k
  T_b.append(detach(U_b,k))

End block:
  commit x_b,0
  update original base cache as usual
  update compressed steering state
  update recent exact memory
  discard full T_b
```

One-step-lag rule:

```text
trace from step k cannot be used to steer the same step k
```

## KV/Attention Concern

Naively reading every full KV tensor can grow expensive. The planned solution is
not to store or attend over all raw KV forever.

Use compressed representations:

- per-layer pooled KV summaries
- low-rank MLA anchors
- Kimi/linear-attention state update
- bounded recent exact token memory
- block-local denoise trace only inside the current block

The base KV path remains unchanged. The steering module reads summaries and
produces residuals; it does not mutate the frozen base cache.

## Suggested Training

Start with SFT only:

```text
base frozen
optimizer sees steering parameters only
zero-init residual heads
residual norm penalty
teacher-forced trace SFT first
online steered trace SFT later
```

Loss:

```text
L = CE + lambda_res * residual_norm
    + optional action_type_loss
    + optional coord_loss
    + optional structure_loss
```

Do not start RL until strict action validity is nontrivial at the target block
size.

## Required Validation

Before claiming steering works:

- frozen-base hash check passes
- trainable parameter names are steering-only
- zero-init identity smoke test passes
- strict JSON improves over base dVLM
- repaired metric is reported separately
- action-type diversity improves
- coordinate range remains norm1000
- latency overhead is measured
- AndroidWorld task progress improves, not only SFT loss

## Non-Claims

This plan does not prove:

- steering already works
- steering solves coordinate mismatch
- steering replaces data curation
- steering fixes bd4/bd8 without noisy-branch training

Safe current claim:

```text
The planned steering module should be KV/cache-aware and residual-only. It will
act as a mode selector over a frozen Fast-dVLM backbone rather than a standalone
policy.
```
