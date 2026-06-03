"""Fast-dVLM dual-stream attention mask (asymmetric vision-efficient layout).

Faithful port of NVlabs/Fast-dLLM `fast_dvlm` modeling
(mask fns + `compute_response_block_idx`).

Layout:  ``[ noisy(L_text) | clean(L_clean) ]`` ; vision tokens are dropped from the
noisy half (vision-efficient concatenation).

Rules:
  * N2N (block_diagonal)      : noisy<->noisy within the same turn, bidirectional.
  * N2C (offset_block_causal) : noisy -> clean of strictly earlier turns (context, incl. vision).
  * C2C (x0_causal)           : clean is plain token-level causal.

Two backends are provided:
  * :func:`build_dense_mask`  -> dense additive float mask for ``sdpa`` (verified on L40S).
  * :func:`build_block_mask`  -> ``torch.nn.attention.flex_attention.BlockMask`` for the
    fused, block-sparse FlexAttention kernel (recommended on H100; ~2-2.5x faster, less mem).
"""
from __future__ import annotations
import torch


def compute_response_block_idx(labels: torch.Tensor, block_size: int):
    """``labels`` [B, L] (-100 = prompt/ignore). Returns ``(response_block_idx[L],
    turn_idx[L], n_blocks)``.

    Each response *segment* is split into blocks of ``block_size`` using ceil division,
    so the last block of every segment is auto-truncated at the response boundary and a
    block never spans two turns.
    """
    labels_single = labels[0]
    seq_len = labels_single.shape[0]
    response_mask = labels_single != -100
    response_block_idx = torch.full((seq_len,), -1, dtype=torch.int64)
    turn_idx = torch.zeros((seq_len,), dtype=torch.int64)
    current_block, in_response, pos_in_seg = 0, False, 0
    for i in range(seq_len):
        if response_mask[i]:
            if not in_response:
                in_response, pos_in_seg = True, 0
            response_block_idx[i] = current_block + (pos_in_seg // block_size)
            pos_in_seg += 1
        elif in_response:
            current_block += (pos_in_seg + block_size - 1) // block_size  # ceil -> auto-truncate
            in_response = False
    for i in range(1, seq_len):
        turn_idx[i] = turn_idx[i - 1] + (1 if response_block_idx[i] != response_block_idx[i - 1] else 0)
    if in_response:
        current_block += (pos_in_seg + block_size - 1) // block_size
    return response_block_idx, turn_idx, current_block


def asymmetric_allowed(q_idx, kv_idx, turn_idx_noisy, turn_idx_clean, n_noisy):
    """Vectorised boolean 'allowed' mask for the ``[noisy | clean]`` layout
    (verbatim rules from Fast-dVLM ``hybrid_block_causal_mask_multiturn_asymmetric``)."""
    x0_q = q_idx >= n_noisy
    x0_kv = kv_idx >= n_noisy
    pos_q = torch.where(x0_q, q_idx - n_noisy, q_idx)
    pos_kv = torch.where(x0_kv, kv_idx - n_noisy, kv_idx)
    tq = torch.where(x0_q, turn_idx_clean[torch.clamp(pos_q, max=turn_idx_clean.shape[0] - 1)],
                     turn_idx_noisy[torch.clamp(pos_q, max=turn_idx_noisy.shape[0] - 1)])
    tk = torch.where(x0_kv, turn_idx_clean[torch.clamp(pos_kv, max=turn_idx_clean.shape[0] - 1)],
                     turn_idx_noisy[torch.clamp(pos_kv, max=turn_idx_noisy.shape[0] - 1)])
    block_diagonal = (~x0_q) & (~x0_kv) & (tq == tk)          # N2N
    offset_block_causal = (tq > tk) & x0_kv & (~x0_q)         # N2C
    x0_causal = x0_q & x0_kv & (pos_q >= pos_kv)              # C2C
    return block_diagonal | offset_block_causal | x0_causal


def stream_meta(labels_clean: torch.Tensor, vision_mask: torch.Tensor, block_size: int):
    """Precompute everything the masks need for one episode (single B=1 sequence)."""
    L_clean = labels_clean.shape[0]
    rbi, turn, n_blocks = compute_response_block_idx(labels_clean.unsqueeze(0).cpu(), block_size)
    text_positions = (~vision_mask).nonzero(as_tuple=True)[0]
    L_text = text_positions.shape[0]
    return dict(text_positions=text_positions, L_text=int(L_text), L_clean=int(L_clean),
                total=int(L_text + L_clean), n_blocks=int(n_blocks),
                response_block_idx=rbi, turn_idx=turn)


def build_dense_mask(meta, turn, device, dtype):
    """Dense additive float mask [1, 1, total, total] for ``F.scaled_dot_product_attention``."""
    tp = meta["text_positions"].to(device)
    Lt, total = meta["L_text"], meta["total"]
    tc = turn.to(device)
    tn = tc[tp]
    qi = torch.arange(total, device=device).view(total, 1).expand(total, total)
    ki = torch.arange(total, device=device).view(1, total).expand(total, total)
    allowed = asymmetric_allowed(qi, ki, tn, tc, Lt)
    add = torch.zeros(total, total, device=device, dtype=dtype)
    return add.masked_fill(~allowed, torch.finfo(dtype).min).view(1, 1, total, total)


def build_block_mask(meta, turn, device, num_heads, batch=1):
    """FlexAttention ``BlockMask`` for the dual-stream pattern (H100 fast path).

    Pass the returned object straight into the model as ``attention_mask`` — transformers'
    ``create_causal_mask`` returns 4D masks (incl. ``BlockMask``) as-is, and the registered
    ``flex_block`` attention interface consumes it. See :mod:`fast_dvlm.flex_attn`.
    """
    from torch.nn.attention.flex_attention import create_block_mask
    tp = meta["text_positions"].to(device)
    Lt, total = meta["L_text"], meta["total"]
    tc = turn.to(device)
    tn = tc[tp]

    def mask_mod(b, h, q_idx, kv_idx):
        return asymmetric_allowed(q_idx, kv_idx, tn, tc, Lt)

    return create_block_mask(mask_mod, B=batch, H=num_heads, Q_LEN=total, KV_LEN=total, device=device)
