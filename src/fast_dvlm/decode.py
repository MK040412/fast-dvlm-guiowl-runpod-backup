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


@torch.no_grad()
def dvlm_generate(model, processor, inputs, gen_len=32, tau=0.9):
    """One-block confidence-parallel diffusion decode of the response. No KV-cache (naive)."""
    tok = processor.tokenizer
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    lm, embed, head = model.model.language_model, model.get_input_embeddings(), model.lm_head
    pid = inputs.input_ids[0]; P = pid.shape[0]; vis = pid == IMG_TOKEN_ID
    pe = embed(pid).unsqueeze(0).clone()
    vo = model.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
    pe[0, vis, :] = torch.cat(list(vo.pooler_output), 0).to(dt)
    total = P + gen_len
    qi = torch.arange(total, device=dev)
    allowed = (qi.view(-1, 1) >= P) | (qi.view(-1, 1) >= qi.view(1, -1))   # gen all, prompt causal
    amask = torch.zeros(total, total, device=dev, dtype=dt).masked_fill(
        ~allowed, torch.finfo(dt).min).view(1, 1, total, total)
    posid = torch.arange(total, device=dev).view(1, 1, total).expand(3, 1, total).contiguous()
    gen = torch.full((gen_len,), MASK_TOKEN_ID, device=dev)
    torch.cuda.synchronize(); t = time.time(); nfe = 0; end_at = None
    for _ in range(gen_len):
        if not (gen == MASK_TOKEN_ID).any():
            break
        cur = torch.cat([pe, embed(gen).unsqueeze(0)], 1).to(dt)
        lg = head(lm(inputs_embeds=cur, attention_mask=amask, position_ids=posid,
                     use_cache=False).last_hidden_state)[0]
        nfe += 1
        masked = (gen == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        prob = F.softmax(lg[P + masked - 1].float(), -1)     # token-shift
        conf, pred = prob.max(-1)
        take = conf > tau
        if take.sum() == 0:
            j = conf.argmax(); gen[masked[j]] = pred[j]
        else:
            gen[masked[take]] = pred[take]
        ie = (gen == IM_END).nonzero()
        if len(ie) and not (gen[:int(ie[0, 0])] == MASK_TOKEN_ID).any():
            end_at = int(ie[0, 0]); break
    torch.cuda.synchronize(); ms = (time.time() - t) * 1000
    n = (end_at + 1) if end_at is not None else gen_len
    return tok.decode(gen[:n], skip_special_tokens=True).strip(), n, nfe, ms
