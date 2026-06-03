"""Fast-dVLM dual-stream multimodal forward + dual CE loss, with optional teacher-KD.

L = 0.5*CE_noisy(2 pairs) + CE_clean(pair-1) [+ kd_weight * KL(student_clean || frozen_AR_teacher)]

The KD term distills the FROZEN autoregressive teacher's next-token distribution into the
student's clean/causal branch on the response tokens. The teacher grounds well in AR mode, so
this transfers grounding (preventing the collapse-to-prior seen with label-only fine-tuning)
while the diffusion branch keeps the block-diffusion (dVLM) capability. Teacher & student share
the frozen vision tower, so vision embeds are reused (no extra ViT forward).

Faithful: per-block Bernoulli noising, complementary pair, vision-efficient concat, auto-
truncated blocks, true vision M-RoPE + DeepStack injection.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .mask import stream_meta, build_dense_mask, build_block_mask

IMG_TOKEN_ID = 151655
IM_END_ID = 151645
MASK_TOKEN_ID = 151665
MIN_NOISE = 1e-3


def dual_stream_loss(model, sample, bd, *, attn="sdpa", min_noise=MIN_NOISE,
                     use_mrope=True, use_deepstack=True,
                     teacher=None, kd_weight=0.0, kd_temp=1.0):
    cfg = model.config.text_config
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    lm, embed, head = model.model.language_model, model.get_input_embeddings(), model.lm_head
    V, Hq = cfg.vocab_size, cfg.num_attention_heads

    ids = sample["input_ids"].to(dev)
    labels = sample["labels"].to(dev)
    L = ids.shape[0]
    vis = ids == IMG_TOKEN_ID
    igt = sample["image_grid_thw"].to(dev)

    # --- vision features (pooler for the embedding merge + deepstack for layer injection) ---
    clean = embed(ids).unsqueeze(0).clone()
    deepstack = None
    if "vemb" in sample and (not use_deepstack or "deepstack_features" in sample):
        img_emb = sample["vemb"].to(dev, dt)
        if use_deepstack and "deepstack_features" in sample:
            deepstack = [d.to(dev, dt) for d in sample["deepstack_features"]]
    else:
        vo = model.model.get_image_features(sample["pixel_values"].to(dev), igt)
        img_emb = torch.cat(list(vo.pooler_output), 0).to(dt)
        if use_deepstack and getattr(vo, "deepstack_features", None) is not None:
            deepstack = [d.to(dt) for d in vo.deepstack_features]
    clean[0, vis, :] = img_emb

    meta = stream_meta(labels, vis, bd)
    rbi = meta["response_block_idx"].to(dev)
    turn = meta["turn_idx"]
    text_pos = meta["text_positions"].to(dev)
    Lt, total, nb = meta["L_text"], meta["total"], meta["n_blocks"]

    resp = labels != -100
    pblk = (1 - min_noise) * torch.rand(max(nb, 1), device=dev) + min_noise
    mask_idx = (torch.rand(L, device=dev) < pblk[rbi.clamp(min=0)]) & (rbi >= 0)
    im_end = (ids == IM_END_ID) & resp
    mask_idx = mask_idx | im_end

    if attn == "flex":
        amask = build_block_mask(meta, turn, dev, Hq, batch=2)
    else:
        amask = build_dense_mask(meta, turn, dev, dt).expand(2, 1, total, total)

    mmt = sample.get("mm_token_type_ids", None)
    opos = None
    if use_mrope and mmt is not None:
        am = torch.ones(1, L, dtype=torch.long, device=dev)
        opos, _ = model.model.get_rope_index(ids.unsqueeze(0), mmt.to(dev).unsqueeze(0), igt, None, attention_mask=am)
        combined = torch.cat([opos[:, :, text_pos], opos], dim=2)
        posid = combined.expand(3, 2, total).contiguous()
    else:
        base = torch.arange(L, device=dev)
        pos = torch.cat([base[text_pos], base])
        posid = pos.view(1, 1, total).expand(3, 2, total).contiguous()

    def make_pair(mi):
        nids = ids.clone(); nids[mi] = MASK_TOKEN_ID
        nemb = embed(nids); nemb[vis] = clean[0][vis]
        demb = torch.cat([nemb[text_pos].unsqueeze(0), clean], dim=1)
        lab = labels.clone(); lab[~mi] = -100
        return demb, lab[text_pos]

    d1, ln1 = make_pair(mask_idx)
    comp = (resp & ~mask_idx) | im_end
    d2, ln2 = make_pair(comp)
    demb = torch.cat([d1, d2], dim=0).to(dt)

    extra = {}
    if deepstack is not None:
        vpm = torch.cat([torch.zeros(Lt, dtype=torch.bool, device=dev), vis]).view(1, total).expand(2, total).contiguous()
        extra["visual_pos_masks"] = vpm
        extra["deepstack_visual_embeds"] = [d.repeat(2, 1) for d in deepstack]

    hidden = lm(inputs_embeds=demb, attention_mask=amask, position_ids=posid,
                use_cache=False, **extra).last_hidden_state

    nlog = head(hidden[:, :Lt, :])
    ln = torch.stack([ln1, ln2], 0)
    ce_n = F.cross_entropy(nlog[:, :-1].reshape(-1, V).float(), ln[:, 1:].reshape(-1), ignore_index=-100)
    clog = head(hidden[:1, Lt:, :])              # student clean/causal branch [1, L, V]
    ce_c = F.cross_entropy(clog[:, :-1].reshape(-1, V).float(), labels[1:].reshape(-1), ignore_index=-100)
    loss = 0.5 * ce_n + ce_c
    info = {"ce_noisy": ce_n.item(), "ce_clean": ce_c.item(), "total": total, "n_blocks": nb}

    # --- teacher-KD: distill the frozen AR teacher into the clean branch on response tokens ---
    if teacher is not None and kd_weight > 0:
        idx = (labels[1:] != -100).nonzero(as_tuple=True)[0]   # positions predicting a response token
        if idx.numel() > 0:
            t_pos = opos if opos is not None else torch.arange(L, device=dev).view(1, 1, L).expand(3, 1, L)
            with torch.no_grad():
                t_clean = teacher.get_input_embeddings()(ids).unsqueeze(0).to(dt)
                t_clean[0, vis, :] = img_emb            # shared frozen vision
                th = teacher.model.language_model(inputs_embeds=t_clean, position_ids=t_pos,
                                                  use_cache=False).last_hidden_state
                t_logits = teacher.lm_head(th[0, idx]).float()    # [n_resp, V]
            s_logits = clog[0, idx].float()
            T = kd_temp
            kd = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1),
                          reduction="batchmean") * (T * T)
            loss = loss + kd_weight * kd
            info["kd"] = kd.item()
    return loss, info
