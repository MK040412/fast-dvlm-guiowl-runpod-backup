# GitHub Backup Scope

This repository backup contains source code, steering adapter implementation, training/eval entrypoints, Vultr AndroidWorld orchestration scripts, task sets, and lightweight JSON/Markdown summaries.

Current decision and handoff documents:

- `docs/CURRENT_MODEL_DECISIONS_2026_06_06.md`
- `docs/COORDINATE_CONVENTION.md`
- `docs/TPU_KD_TRAINING_NOTES.md`
- `docs/ANDROIDWORLD_LOCAL_STATUS.md`
- `docs/DATA_CURATION_PLAN.md`
- `docs/DATA_CURATION_PLAN_FULL.md`
- `docs/STEERABLE_MODULE_PLAN.md`
- `docs/STEERABLE_MODULE_PLAN_FULL.md`
- `docs/KNOWN_FAILURES_AND_NON_CLAIMS.md`
- `docs/CLAUDE_HANDOFF_2026_06_06.md`
- `docs/LOCAL_GMAIL_DECISIONS_2026_06_06.md`

Excluded by design:

- API tokens, passwords, SSH keys, and machine credentials
- model weights and adapter safetensors
- AITW/AndroidWorld datasets
- Android SDK and AVD images
- raw AndroidWorld `.pkl.gz` episodes, screenshots, videos, and large logs

Large artifacts should be uploaded to Hugging Face. The current steering adapter has already been uploaded to:

`KMK040412/fast-dvlm-guiowl-bard-bd32/steering_sft/bd32_steer_sft_300m_bf16`
