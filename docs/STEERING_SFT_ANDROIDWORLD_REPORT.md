# Fast-dVLM GUI-Owl Steering SFT and AndroidWorld Eval Report

This repository snapshot documents the current Fast-dVLM / GUI-Owl work on RunPod and the AndroidWorld benchmark setup on Vultr bare metal. It intentionally excludes model weights, tokens, passwords, AVD images, Android SDK files, dataset parquet files, and large benchmark artifacts.

## Model State

- Frozen base checkpoint: `/workspace/dvlm_ckpts/ckpt_bard_bd32`
- Base/original GUI-Owl model: `/workspace/models/GUI-Owl-1.5-2B-Instruct`
- Steering adapter checkpoint: `/workspace/dvlm_ckpts/ckpt_bard_bd32_steer_sft`
- Hugging Face adapter upload path: `KMK040412/fast-dvlm-guiowl-bard-bd32/steering_sft/bd32_steer_sft_300m_bf16`

The steering adapter is adapter-only. It does not include base model weights.

## Steering Architecture

The added steering package is under `src/fast_dvlm/steering/`.

Main components:

- `TraceBank`: stores block-local denoise traces.
- `TraceEncoder`: projects hidden fallback traces, current step features, step embeddings, and position embeddings.
- `KimiTraceSteering`: SSD/MLA-style steering module.
- `ResidualInjector`: injects residuals into selected frozen language layers.
- `SteeringWrapper`: orchestrates trace capture and residual preparation.
- `json_repair.py` and `metrics.py`: structural output parsing and metric helpers.

The base model remains frozen. Trainable parameters are restricted to steering modules and residual heads. No LoRA, tokenizer changes, embedding training, or LM head training are used.

Current trace source is `hidden_fallback`, not true K/V capture.

## Steering SFT Training Snapshot

Main training output:

- `/workspace/dvlm_ckpts/ckpt_bard_bd32_steer_sft/steering_model.safetensors`
- Adapter size: about 565 MB bf16
- Trainable parameters: about 296M
- Final run: 644 optimizer steps, 5,145 samples
- Training log: `/workspace/dvlm_ckpts/ckpt_bard_bd32_steer_sft/train_log.jsonl`

Validation:

- Frozen check passed: `/workspace/paper_logs/steering_frozen_check_300m.json`
- Zero-init identity smoke passed: `/workspace/paper_logs/steering_identity_smoke_300m.json`
- Base gradient nonzero count stayed 0 in training logs.

Gradient interpretation:

The console value `grad=0.000` was a formatting artifact. JSON logs show nonzero steering gradient on every logged step. Final run gradient range was approximately `4.08e-05` to `5.16e-01`, mean about `3.20e-03`. Residual projection heads moved away from zero; residual head nonzero fraction was 1.0.

Caveat: the gradient signal is small for many steps, so the adapter did update, but this does not prove that the full 300M steering body learned a strong policy correction. Paired evaluation is required.

## AndroidWorld Infrastructure

Vultr bare metal setup:

- CPU: AMD EPYC 7443P, 24 cores / 48 threads
- RAM: 251 GiB
- KVM available
- Android SDK root: `/data2/android-sdk`
- AVD root: `/data2/android-avd`
- AndroidWorld eval root: `/data2/androidworld_eval`

The stable emulator image is API 33 Google APIs x86_64. Android 35 rejected older AndroidWorld APKs, and Android 30 lacked `POST_NOTIFICATIONS`; API 33 is the working compromise.

Eight emulators were started and validated:

- console/grpc: `5554/8554`, `5556/8555`, `5558/8556`, `5560/8557`, `5562/8558`, `5564/8559`, `5566/8560`, `5568/8561`

RunPod policy servers:

- base AR replicas: ports `8123`, `8125`, `8127`, `8129`
- bd32 dVLM replicas: ports `8124`, `8126`, `8128`, `8130`

The Vultr machine reaches the RunPod policy servers through reverse SSH tunnels.

## Benchmark Runner

The Vultr benchmark scripts are included under `scripts/androidworld_vultr/`.

Important scripts:

- `prepare_avds.sh`
- `start_emulators.sh`
- `setup_androidworld_envs.sh`
- `run_eval_sharded_8lane.sh`
- `run_large_benchmarks_8lane.sh`
- `summarize_androidworld_run.py`

Task sets:

- `task_sets/general_core.txt`
- `task_sets/standard_smoke.txt`

Large benchmark root on Vultr:

- `/data2/androidworld_eval/runs/large_8lane_20260603_213633`

Execution plan:

- `general_core`: 24 tasks, 4 shards per model, 8 workers total
- `standard_full`: 116 AndroidWorld tasks, 4 shards per model, 8 workers total
- `N_TASK_COMBINATIONS=1`
- `TASK_RANDOM_SEED=30`

The benchmark stores AndroidWorld `.pkl.gz` episode checkpoints. Each episode contains task success, step-level raw model output, strict JSON parsing metadata, repair metadata, Android action, and latency fields.

## General-Core Snapshot

Copied summary: `docs/eval_results/general_core_summary.json`

General-core finished with 24 episodes per model:

| model | episodes | success_rate | strict_json_rate_mean | mobile_use_rate_mean | repair_rate_mean | model_latency_ms_mean |
|---|---:|---:|---:|---:|---:|---:|
| base AR | 24 | 0.0 | 0.9962 | 0.8472 | 0.3441 | 1220.79 |
| bd32 dVLM | 24 | 0.0 | 0.7940 | 0.9622 | 0.1682 | 1088.72 |

Important caveat: this is an infrastructure validation result, not a final paper claim. Success rate is currently 0 for both models on this snapshot, and `current` means bd32 dVLM, not the steering adapter.

## Current Benchmark Status

At the time of this repository snapshot:

- `general_core` completed and produced summary files.
- `standard_full` had started and was running with 8 workers.
- Final benchmark artifacts should be uploaded to Hugging Face, not committed to GitHub.

## Next Required Work

1. Let `standard_full` finish.
2. Upload benchmark result directory to Hugging Face.
3. Add steering-adapter inference support to `guiowl_action_server.py`.
4. Re-run the same benchmark as:
   - base GUI-Owl AR
   - bd32 dVLM strict/repair
   - bd32 + steering strict/repair
5. Report strict and repaired metrics separately.
6. Add seed/task-combination repetitions before making final paper claims.
