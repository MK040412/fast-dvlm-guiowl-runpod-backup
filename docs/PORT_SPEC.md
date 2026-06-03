# Fast-dVLM (Fast-dLLM-v2 multimodal) → Qwen3-VL (GUI-Owl-1.5-2B) Direct-Conversion Spec
# Extracted by subagent from arXiv 2604.06832 + NVlabs/Fast-dLLM code + HF remote modeling. 2026-06-03.

## Source-of-truth map
- Local clone: /tmp/Fast-dLLM (re-cloneable: github.com/NVlabs/Fast-dLLM)
- Collator (single clean stream): third_party/lmflow/datasets/multi_modal_dataset.py  DataCollatorForQwenVL L372-505
- Training loop: third_party/lmflow/pipeline/finetuner.py (stock HF Trainer; NO mdm logic)
- **Dual-stream+mask+loss = HF remote modeling.py** of `Efficient-Large-Model/Fast_dVLM_3B` (saved /tmp/fastdvlm_modeling.py, 2300+ lines), forward L1714-2043.
- Canonical text-only mask: `Fast_dLLM_v2_7B/modeling.py` (saved /tmp/v2_modeling.py) block_diff_mask L44-87.
- Entry: fast_dvlm/train_scripts/finetune_dvlm.py

KEY FACT: collator emits ONE clean [vision+full text] stream w/ assistant-only labels. The MODEL.forward does ALL mdm: noising, [noisy|clean] doubled stream, complementary pair (doubled along BATCH), flex-attention BlockMask, dual CE. Port = reproduce Fast_dVLMForConditionalGeneration.forward against Qwen3-VL.

Constants (Fast_dVLM_3B): mask_id=151665 (|<MASK>|), im_start=151644, im_end=151645, assistant_target=[151644,77091,198] (<|im_start|>assistant\n), image_token_id=151655, vision_start=151652, vision_token=151654, video=151656. Fast-dVLM is Qwen2.5-VL: mrope_section [16,24,24], head_dim128, 16q/2kv, 36L, hidden2048, bd_size top=32/text=8, minimum_noise_level=1e-3, tie_word_embeddings, NO QK-norm.

## 1. STREAM CONSTRUCTION
1a. Collator builds ONE clean stream + labels (mask all except assistant spans by scanning target_sequence [151644,77091,198]; response_end includes <|im_end|>\n). 2D attention_mask from collator is IGNORED in training (model overwrites w/ 4D flex BlockMask).
1b. Inside model.forward: clean = full [vision+text] unchanged; noisy = response tokens stochastically -> mask_id, VISION TOKENS DROPPED entirely (vision-efficient). Doubled layout per pair: [noisy(L_text) | clean(L)] cat dim=1. Complementary pair cat dim=0 -> [2B, L_text+L].
1c. Multimodal noising (fastdvlm_modeling L1827-1961):
  vision_token_mask = (ids==image_token)|(ids==video)|(ids==vision_start)
  response_block_idx,turn_idx,n_blocks = compute_response_block_idx(labels, bd_size)
  response_mask = labels!=-100 ; eps=1e-3 ; t=rand(n_blocks); p_mask_per_block=(1-eps)*t+eps  (one noise level PER BLOCK)
  mask_indices[:,i] = rand(B) < p_mask_per_block[block_i]   for i with block_i>=0
  im_end in response ALWAYS masked: mask_indices |= (ids==im_end)&response_mask
  noisy_input_ids[mask_indices]=mask_id ; noisy_embeds = where(vision, original_embeds, embed(noisy_ids))
  labels_noisy[~mask_indices]=-100   (loss only on masked response tokens)
  text_positions = (~vision_token_mask[0]).nonzero ; L_text=len ; compress noisy_* to text_positions
  combined_position_ids = cat([noisy_pos, original_pos], dim=2)  # [3,B,L_text+L]
1d. Complementary pair: complementary_mask_indices = response_mask & ~mask_indices (| im_end); same vision-keep+compress; labels keep masked positions only. Stack pair1,pair2 along BATCH -> [2B,...]. self._noisy_seq_len=L_text.
  WARN: text_positions from row 0 only -> unsafe for heterogeneous vision counts; bucket batch by image count or recompute per-row.

## 2. ATTENTION MASK (flex_attention BlockMask, NOT additive 4D float)
2a. Canonical (v2_modeling L44-87), layout [xt(0..n)|x0(n..2n)]:
  x0_flag = idx>=n ; block = (idx-n)//bs if x0 else idx//bs
  M_BD (N2N): block_q==block_kv & x0_flag_q==x0_flag_kv           (bidir within same-side block)
  M_OBC(N2C): block_q>block_kv & x0_kv==1 & x0_q==0               (noisy->strictly earlier clean blocks)
  M_BC (C2C): block_q>=block_kv & x0_kv==1 & x0_q==1              (clean block-causal incl own)
  mask = M_BD | M_OBC | M_BC
2b. Fast-dVLM multiturn (fastdvlm_modeling L34-124) uses PER-TOKEN response_block_idx/turn_idx (prompt causal, each response blocks independently). Asymmetric variant (vision-efficient): n_noisy=L_text, separate turn_idx_noisy(compressed)+turn_idx_clean(full), pick turn table per side via x0_flag. Rules same.
2c. compute_response_block_idx (L1623-1680): prompt=-1; within a response segment block_in_segment=response_pos_in_segment//block_size; at segment end current_block += CEIL(pos/block_size) -> **AUTO-TRUNCATION** (last block shorter, never crosses response boundary). turn_idx increments where block idx changes.
2d. bd_size (annealed) -> block_size -> regroups response tokens; mask regenerated every forward.
2e. Eval: overrides attention_mask w/ eval_block_diff_mask or eval_causal_mask (dense bool [Q,KV]).

## 3. POSITION IDS / RoPE (trickiest)
- Both halves SHARE clean position_ids. Multimodal: get_rope_index on ORIGINAL pre-doubled vision-aware seq, then combined=cat([pos[:,:,text_positions], pos], dim=2). RoPE applied SEPARATELY per half (split at noisy_seq_len=L_text): q_1=q[:,:,:nl], q_2=q[:,:,nl:]; cos_1/sin_1 vs cos_2/sin_2; apply_multimodal_rotary_pos_emb each. noisy_seq_len threaded via kwargs, popped before FA backend.
- mrope (Qwen2.5 split-concat): mrope_section*2 then cat([m[i%3] for split]). **Qwen3-VL uses INTERLEAVED (apply_interleaved_rope) + mrope_section [24,20,20] + rope_theta 5e6** -> REPLACE.

## 4. LOSS (dual CE + token shift) fastdvlm_modeling L2013-2031
  noisy_len=_noisy_seq_len
  logits = lm_head(hidden[:, :noisy_len, :])                 # noisy half of BOTH pairs [2B,L_text]
  loss  = loss_function(logits, labels) * 0.5                 # diffusion branch, *0.5
  causal_hidden = hidden[:B, noisy_len:, :]                   # clean half of pair1 ONLY [B,L]
  loss += loss_function(lm_head(causal_hidden), original_labels)   # causal branch weight 1.0
  => L = 0.5*CE_noisy(2B) + CE_clean(B)  (effective alpha=beta=0.5 per paper)
  token shift = inside HF ForCausalLMLoss (logits[...,:-1] predict labels[...,1:]); pass labels UNSHIFTED.
  pass num_items_in_batch*2 to keep token-mean norm correct.

## 5. BLOCK-SIZE ANNEALING fastdvlm_modeling L1758-1767
  update_ratio = kwargs.get('update_ratio',1.0)  # u in [0,1] = step/total; DEAD CODE in release (always 1.0!)
  max_power=int(log2(bd_size)) ; possible=[2**i for i in range(2,max_power+1)]  # bd_size32 -> {4,8,16,32}
  scaled=sqrt(update_ratio) ; idx=min(int(scaled*len(possible)), len-1) ; bd_size=possible[idx]
  => MUST subclass Trainer to inject update_ratio=global_step/max_steps into model inputs (else no annealing).

## 6. MASK -> MODEL
  No _update_causal_mask, no additive 4D, no monkeypatch. Custom TextModel.forward passes attention_mask (=BlockMask) straight to layers; Attention.forward when self.training calls fused_flex_attention(q,k,v, mask=BlockMask) = flex_attention(..., block_mask=, enable_gqa=True). Port: subclass Qwen3VLTextAttention to branch on self.training into flex_attention; override Qwen3VLTextModel.forward to NOT call _update_causal_mask and pass BlockMask through + split RoPE per half via noisy_seq_len.

## 7. QWEN2.5-VL -> QWEN3-VL PORT DELTAS
1. Modules at model.language_model.{layers,embed_tokens}; classes Qwen3VLTextAttention/DecoderLayer/TextModel/ForConditionalGeneration.
2. Subclass Qwen3VLTextAttention: add self.training flex branch + per-half RoPE split.
3. **QK-NORM NEW**: Qwen3-VL has per-head q_norm/k_norm (RMSNorm head_dim128) applied AFTER reshape-to-heads, BEFORE RoPE, on BOTH halves identically. Fast-dVLM had none.
4. **mRoPE interleaved**: replace apply_multimodal_rotary_pos_emb w/ Qwen3-VL apply_interleaved_rope; keep per-half split.
5. mrope_section [24,20,20] from config.rope_scaling (not hardcode). rope_theta 5e6.
6. Vision ids from config (image_token_id etc.); Qwen3-VL DeepStack vision merge (deepstack_visual_indexes) -> use stock Qwen3-VL vision path for CLEAN half (vision dropped in noisy half anyway).
7. Port Qwen3-VL get_rope_index (3D mrope); keep combined=cat([pos[:,:,text_positions],pos],dim=2).
8. BYPASS Qwen3VLTextModel._update_causal_mask (override forward) — flex needs BlockMask not float.
9. tie_word_embeddings: check GUI-Owl config (2B typically ties). Use model.lm_head both branches.
10. ForCausalLMLoss default (keeps shift); noisy branch num_items_in_batch*2.
11. GUI-Owl 28L/2048/16q/8kv (group2); enable_gqa=True handles ratio.
12. Add |<MASK>| token (reuse 151665 if free) -> resize embeddings +1; this row is trained.
13. minimum_noise_level=1e-3, p=(1-eps)*U(0,1)+eps per block, im_end always masked.

## RISKS
- update_ratio annealing is dead code (always 1.0) -> custom Trainer required; candidate set in code is {4,8,16,32} w/ sqrt(u) warp.
- text_positions from row0 only -> bucket by image count / recompute per row.
- clean branch uses pair1 only [:B]; *0.5 on noisy[2B] balances. Don't double-count.
- flex_attention needs torch>=2.5 (have 2.7.1 OK) + may need specific GPU; verify on L40S.
