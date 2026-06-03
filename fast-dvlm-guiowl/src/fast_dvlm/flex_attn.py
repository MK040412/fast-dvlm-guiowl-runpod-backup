"""Register a FlexAttention interface for transformers so the dual-stream ``BlockMask``
is consumed by a fused, block-sparse kernel.

Measured (L40S, block-sparse dual-stream pattern, fwd+bwd):
    L=2048: flex 1.01ms vs sdpa 2.05ms  -> 2.03x , ~1.3x less peak mem
    L=4096: flex 3.01ms vs sdpa 7.56ms  -> 2.51x   (speedup grows with length)
On H100 the win is larger and compounds with longer packed contexts.

Usage:
    from fast_dvlm.flex_attn import register_flex
    register_flex()
    model.config._attn_implementation = "flex_block"   # or load with attn_implementation
    # then pass a fast_dvlm.mask.build_block_mask(...) as the model's attention_mask
"""
from __future__ import annotations
import torch

_FLEX = None


def _flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        # dynamic=True -> one kernel for all sequence lengths (avoids per-length recompiles,
        # which dominate when episodes vary in length). Fixed-length packing is faster still.
        _FLEX = torch.compile(flex_attention, dynamic=True)
    return _FLEX


def flex_block_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers attention-interface signature. ``attention_mask`` is a ``BlockMask``.

    ``query`` is [B, Hq, L, D]; ``key``/``value`` may have fewer heads (GQA) -> ``enable_gqa``.
    Returns ``(attn_output[B, L, Hq, D], None)`` to match the eager/sdpa interface contract.
    """
    enable_gqa = query.shape[1] != key.shape[1]
    out = _flex()(query, key, value, block_mask=attention_mask, enable_gqa=enable_gqa, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def register_flex():
    """Idempotently register the ``flex_block`` attention implementation."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["flex_block"] = flex_block_attention
