# Results Preserved In This Backup

All current-model runs used:

- model: `/workspace/dvlm_ckpts/ckpt_bard_bd32`
- decode: `dvlm`
- prompt mode: `mobileagent`
- coordinate mode: `normalized`
- structural repair: enabled
- runner: 8 Android emulator lanes on Vultr, HTTP model servers on RunPod

## Base AR Sanity Check

Normalized coordinate mode made the benchmark harness register task-level
success for the original GUI-Owl AR model:

| Run | Model | Episodes | Success | Strict JSON | Mobile Use | Repair |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mobileagent_norm_smoke_20260603_223919` | base AR | 4 | 2 | 0.975 | 0.975 | 0.000 |
| `mobileagent_norm_smoke_20260603_223919` | bd32 dVLM | 4 | 0 | 0.025 | 0.650 | 0.625 |
| `mobileagent_norm_record2_20260603_224545` | base AR | 4 | 1 | 1.000 | 1.000 | 0.000 |
| `mobileagent_norm_record2_20260603_224545` | bd32 dVLM | 4 | 0 | 0.100 | 0.700 | 0.600 |

This means the benchmark stack itself was capable of detecting task success.

## bd32 dVLM General Runs

| Run | Episodes | Success | Strict JSON Mean | Mobile Use Mean | Repair Mean | Model Latency Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_general_core_sweep_20260603_230159` | 24 | 0 | 0.0476 | 0.7087 | 0.6612 | 4207 ms |
| `current_general_focus_sweep_20260603_230905` | 52 | 0 | 0.0484 | 0.6983 | 0.6505 | 4628 ms |
| `current_general_focus_multicombo_20260603_232318` | 156 | 0 | 0.0243 | 0.6706 | 0.6485 | 4647 ms |

Interpretation:

- Structural repair often recovered a syntactically usable `mobile_use` action.
- Task-level success remained zero in general-focused AndroidWorld tasks.
- The failure mode is therefore not only malformed JSON. The model is still weak
  at action choice, coordinates, and multi-step state tracking under AndroidWorld.

## Interrupted Standard Full Run

The standard full sweep was intentionally stopped when the priority changed back
to bd32 training.

| Run | Episodes Completed | Success | Strict JSON Mean | Mobile Use Mean | Repair Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_standard_full_sweep_20260604_000539` | 10 / 116 | 0 | 0.0232 | 0.7265 | 0.7033 |

This run is partial and should not be reported as a completed standard
benchmark.

## Artifact Locations

GitHub backup:

```text
vultr_androidworld_backup/raw/runs/
```

Hugging Face artifact dataset:

```text
https://huggingface.co/datasets/KMK040412/fast-dvlm-guiowl-androidworld-artifacts
```

The HF upload includes videos and heavier trajectory artifacts that are not
stored in GitHub.

