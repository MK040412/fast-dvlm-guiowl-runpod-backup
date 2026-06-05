# Fast-dVLM / GUI-Owl-1.5-2B — Decisions Log (as of 2026-06-06)

## 1. Coordinate convention (LOCKED)
- Canonical = **`guiowl_norm1000_xy`**: `coordinate:[x,y]`, range 0..1000, origin top-left. Pixels FORBIDDEN in SFT target_json; executor alone maps norm1000→device pixels.
- Verified on TPU: GUI-Owl 1.5, base bd32, `data.py`, AndroidWorld executor all use norm1000.
- **CONFIRMED BUG**: prior Gmail SFT trained on PIXEL coords. `train_fastdvlm_tpu.py::native_action` returns `target_json` verbatim (no normalization); the gmail parquet `target_json` was pixels (e.g. click [204,286] on 412×732). Actual KD mix = 486,296 pixel gmail rows + 268 norm1000 general rows → model learned pixels → AndroidWorld grounding failure (gmail ckpt 0/4; base 1/4).

## 2. Corrected dataset (DONE)
- HF (private) `KMK040412/aitw-androidworld-overfit-mix`: 55,345 ep / 476,043 steps, ALL coords norm1000 verified (0 OOR, 0 parse fail). Gmail bumped to 8k (dedup, no duplication). On TPU at `~/data/aitw_androidworld_overfit_mix_norm1000`.
- Strategy = targeted overfit on AndroidWorld-core tasks (Wifi/Bluetooth/Contacts/Clock) + support (Search/general/web_shopping/Gmail demo/Settings). type ~8%, click:swipe 2.6:1, click spatial grid coverage 99.8%.

## 3. Current LIVE training run (baseline)
- `~/tpu_fastdvlm_runs/kd_aw_overfit_norm1000_full_pad480_cap96_bs32_20260605_171137`, PID 178527.
- From **pixel** `ckpt-bard-bd32-gmail-adb-vitlora-e1-final` (NOT the cleaner 0-1000 base bd32).
- Self-distill loss = 1.0·CE_noisy + 0.75·CE_clean + 0.25·KD_noisy (teacher = own clean branch, stop_grad, temp 2.0). bd-schedule 4/8/16/32@.25. **lr=1e-6 CONSTANT (no warmup/decay)**, adamw_bf16, wd 0, bs 32, 1 epoch (~14,876 steps). pad480/ctx480/cap96/vision_pad96 (HBM-forced from 512/128). mrope/deepstack exact ON. Uploads every 3000 → `KMK040412/fast-dvlm-guiowl-kd-tpu`.
- Health: loss 5.0→2.44 @ step 2009, ce_clean ~0.60 (plateauing), trunc 0, no NaN/OOM.

## 4. Recipe decision (NEXT run) — resource-aware, SOTA-aligned
- SOTA SFT (LLaVA/Qwen-VL/LLaDA) = peak lr **1e-5–2.5e-5 + warmup + cosine/linear decay**, batch 128–1280. Ours (1e-6 flat, bs32) is ~3–9× low after batch-scaling AND missing warmup/decay.
- Constraint: v6e-4 HBM tight (peak 27.3/33.5GB). Free levers (no HBM cost): warmup+cosine, higher lr, wd 0.1, **grad-accum** (key — large effective batch without HBM → unlocks SOTA lr).
- **Next-run recipe**: warmup 100–300 → **peak 1e-5** → cosine→1e-6; **grad-accum ×4–8 (eff batch 128–256)**; wd 0.1; keep pad480/cap96 (ensure coord tokens not truncated); target oversample 2–3× (real overfit).
- **Distillation**: keep 2B self-distill (resource-optimal, no separate teacher). Fixes: (a) teacher-correctness via clean base bd32 OR KD warmup (KD 0 first N steps until ce_clean low); (b) **bd-dependent KD weight** (bd4 ~0.1 → bd32 ~0.4) since kd_noisy plateaus worst at bd32.

## 5. Apply-from-step-3000 plan (agreed, data-driven)
- At step 3000 checkpoint, run `eval_risk_suite.py` → check output coords are norm1000 (not pixel-compressed x≤412) + strict-JSON across bd 1/4/8/16/32.
- **If coords progressing toward 0-1000 (likely)**: resume FROM step-3000 with the new recipe (saves ~1.4h). Optimizer state resets → short warmup covers cold start; cosine over remaining steps.
- **If coords completely stuck pixel**: restart from clean **base bd32** + new recipe.
- **If already perfect**: let current run finish, or bump lr to accelerate.

## 6. Open audit risks (from 3-way codex+claude audit)
1. lr=1e-6 likely too low to overwrite pixel prior in 1 epoch (ce_clean 0.60 plateau = the tell). → §4/§5.
2. Only 4 AW-core tasks covered = "4 smoke + demo", NOT 116-task AndroidWorld. → broad-corpus curation (§7).
3. Trainer has NO coordinate-validation gate (handoff required it). → `validate_coords.py` prepared; wire into next run.

## 7. NEXT data-curation direction (Mobile-Agent-v3.5 / GUI-Owl-1.5 style) — proposed by codex
- Shift from "AiTW overfit" to a BROAD mobile SFT corpus: **AndroidControl (highest ROI) + AMEX + GUI-Odyssey + OpenMobile + grounding (ScreenSpot/RICO/MobileViews)**, all canonicalized to norm1000 mobile_use, with quality scoring, dedup/caps, AndroidWorld leakage control, reserved reasoning/KD columns. Output HF `KMK040412/mobile-gui-sft-mobileagent35-style-v1` + mixes by training-hour budget (5h/10h/20h/40h) + reports.
- Relationship to §2: COMPLEMENTARY — aw-overfit-mix = fast demo/4-task path; broad corpus = real AndroidWorld generalization. Likely train broad (base capability) then sharpen on overfit/target mix.
- **Steering co-training (planned)**: will add the KimiTraceSteering module (TraceBank→TraceEncoder→SSDLinear/MLA→residual injection, mode-selection gate) and co-train alongside SFT. Requires trajectory `history` preserved in the data (codex schema includes it; GUI-Odyssey/OpenMobile multi-step + recovery trajectories are ideal trace material). Steering = mode-selector (doesn't touch denoise); freeze_base optional.
- **Resource reconcile**: this curation runs ON Vultr CPU → **do NOT shut Vultr down if pursuing this plan.** Start with AndroidControl-only 5h mix, validate coords per-source, then expand.
