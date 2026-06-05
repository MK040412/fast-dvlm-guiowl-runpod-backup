"""Inference: AR (causal) generation vs dVLM block-diffusion decoding (single device, B=1).

The dVLM decoder fills the response region with confidence-threshold parallel unmasking
(token-shift: position i predicted from hidden i-1). The single-user latency win is
``tokens / NFE`` (forwards saved); with a KV-cache for the clean prefix it is realised as
wall-clock speedup. Measured on an overfit model: ~2.9x fewer forwards, ~2.8x faster wall
clock even WITHOUT a KV-cache (paper reports ~1.95-2.63 tok/NFE on real benchmarks, up to
6.18x with the SGLang + FP8 serving stack).
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F

from .forward import IMG_TOKEN_ID, MASK_TOKEN_ID
from .mask import build_dense_mask  # noqa: F401  (kept for parity / extension)

IM_END = 151645


@torch.no_grad()
def ar_generate(model, processor, inputs, max_new=32):
    tok = processor.tokenizer
    torch.cuda.synchronize(); t = time.time()
    g = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(); ms = (time.time() - t) * 1000
    new = g[0][inputs.input_ids.shape[1]:]
    end = (new == IM_END).nonzero()
    n = int(end[0, 0]) + 1 if len(end) else len(new)
    return tok.decode(new[:n], skip_special_tokens=True).strip(), n, ms


def _vision_feats(model, inputs, dt):
    """Return (pooler_feats[num_vis, hidden], deepstack_list | None), robust across
    transformers versions. get_image_features may return an object with .pooler_output /
    .deepstack_features, or a tuple (image_embeds, deepstack_image_embeds)."""
    vo = model.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
    deepstack = None
    if isinstance(vo, (tuple, list)) and len(vo) == 2 and isinstance(vo[0], (tuple, list)):
        vfeat = torch.cat(list(vo[0]), 0)
        if vo[1] is not None:
            try:
                deepstack = [d.to(dt) for d in vo[1]]
            except TypeError:
                deepstack = None
    elif hasattr(vo, "pooler_output"):
        vfeat = torch.cat(list(vo.pooler_output), 0)
        ds = getattr(vo, "deepstack_features", None)
        deepstack = [d.to(dt) for d in ds] if ds else None
    elif isinstance(vo, (tuple, list)):
        vfeat = torch.cat([x for x in vo if torch.is_tensor(x)], 0)
    elif torch.is_tensor(vo):
        vfeat = vo
    else:
        raise TypeError(f"unsupported image feature output type: {type(vo)!r}")
    return vfeat.to(dt), deepstack


def _prompt_mrope(model, pid, image_grid_thw, dev):
    """Vision M-RoPE position ids [3,1,P] for the prompt (matches forward.py training);
    None if unavailable so callers fall back to arange."""
    try:
        am = torch.ones(1, pid.shape[0], dtype=torch.long, device=dev)
        opos, _ = model.model.get_rope_index(pid.unsqueeze(0), image_grid_thw, None, am)
        return opos.to(dev)
    except Exception:
        return None


@torch.no_grad()
def _dvlm_core(model, processor, inputs, gen_len, tau, block_size, grounded):
    """Block-wise confidence-parallel diffusion decode (AR over blocks of `block_size`,
    diffusion within a block; token-shift). When grounded=True, injects vision M-RoPE
    position ids + DeepStack features so decode matches training (forward.py) — this is
    required for coordinate accuracy. Returns (text, tokens, nfe, ms)."""
    tok = processor.tokenizer
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    lm, embed, head = model.model.language_model, model.get_input_embeddings(), model.lm_head
    pid = inputs.input_ids[0]; P = pid.shape[0]; vis = pid == IMG_TOKEN_ID
    vfeat, deepstack = _vision_feats(model, inputs, dt)
    if not grounded:
        deepstack = None
    pe = embed(pid).unsqueeze(0).clone(); pe[0, vis, :] = vfeat
    opos = _prompt_mrope(model, pid, inputs.image_grid_thw, dev) if grounded else None

    committed, nfe, stop = [], 0, False
    torch.cuda.synchronize(); t = time.time()
    while not stop and len(committed) < gen_len:
        c = len(committed); A = min(block_size, gen_len - c)
        gen = torch.full((A,), MASK_TOKEN_ID, device=dev)
        ct = torch.tensor(committed, device=dev, dtype=torch.long) if committed else None
        total = P + c + A
        qi = torch.arange(total, device=dev)
        allowed = (qi >= (P + c)).view(-1, 1) | (qi.view(1, -1) <= qi.view(-1, 1))
        amask = torch.zeros(total, total, device=dev, dtype=dt).masked_fill(
            ~allowed, torch.finfo(dt).min).view(1, 1, total, total)
        extra = {}
        if opos is not None:
            start = int(opos.max().item()) + 1
            tail = torch.arange(start, start + c + A, device=dev).view(1, 1, -1).expand(3, 1, c + A)
            posid = torch.cat([opos, tail], dim=2).contiguous()
            if deepstack is not None:
                extra["visual_pos_masks"] = torch.cat(
                    [vis, torch.zeros(c + A, dtype=torch.bool, device=dev)]).view(1, total)
                extra["deepstack_visual_embeds"] = deepstack
        else:
            posid = torch.arange(total, device=dev).view(1, 1, total).expand(3, 1, total).contiguous()
        for _ in range(A):
            if not (gen == MASK_TOKEN_ID).any():
                break
            parts = [pe] + ([embed(ct).unsqueeze(0)] if ct is not None else []) + [embed(gen).unsqueeze(0)]
            cur = torch.cat(parts, 1).to(dt)
            lg = head(lm(inputs_embeds=cur, attention_mask=amask, position_ids=posid,
                         use_cache=False, **extra).last_hidden_state)[0]
            nfe += 1
            masked = (gen == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            prob = F.softmax(lg[P + c + masked - 1].float(), -1)     # token-shift
            conf, pred = prob.max(-1)
            take = conf > tau
            if take.sum() == 0:
                j = conf.argmax(); gen[masked[j]] = pred[j]
            else:
                gen[masked[take]] = pred[take]
            ie = (gen == IM_END).nonzero()
            if len(ie) and not (gen[: int(ie[0, 0])] == MASK_TOKEN_ID).any():
                committed += gen[: int(ie[0, 0]) + 1].tolist(); stop = True; break
        if not stop:
            committed += gen.tolist()
    torch.cuda.synchronize(); ms = (time.time() - t) * 1000
    n = committed.index(IM_END) + 1 if IM_END in committed else len(committed)
    text = tok.decode([x for x in committed[:n] if x != IM_END], skip_special_tokens=True).strip()
    return text, n, nfe, ms


@torch.no_grad()
def dvlm_generate(model, processor, inputs, gen_len=64, tau=0.9, block_size=32, grounded=True):
    """Faithful BD3-LM block-diffusion decode. Defaults: trained block_size=32, grounded
    (M-RoPE + DeepStack injected to match forward.py). Set grounded=False for the legacy
    pooler-only + arange decode, or block_size=gen_len for a single-block decode."""
    return _dvlm_core(model, processor, inputs, gen_len, tau, block_size, grounded)


# Back-compat alias
dvlm_generate_blockwise = dvlm_generate
