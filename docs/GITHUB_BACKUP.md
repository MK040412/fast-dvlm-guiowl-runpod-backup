# GitHub Backup Scope

This repository backup contains source code, steering adapter implementation, training/eval entrypoints, Vultr AndroidWorld orchestration scripts, task sets, and lightweight JSON/Markdown summaries.

Excluded by design:

- API tokens, passwords, SSH keys, and machine credentials
- model weights and adapter safetensors
- AITW/AndroidWorld datasets
- Android SDK and AVD images
- raw AndroidWorld `.pkl.gz` episodes, screenshots, videos, and large logs

Large artifacts should be uploaded to Hugging Face. The current steering adapter has already been uploaded to:

`KMK040412/fast-dvlm-guiowl-bard-bd32/steering_sft/bd32_steer_sft_300m_bf16`
