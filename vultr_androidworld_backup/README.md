# Vultr AndroidWorld Backup

This directory preserves the AndroidWorld evaluation setup that was running on
the Vultr bare-metal host before teardown.

No API tokens, passwords, or private SSH keys are included.

## What Is Backed Up

- `raw/guiowl_androidworld_agent.py`: GUI-Owl/Fast-dVLM AndroidWorld agent.
- `raw/run_guiowl_androidworld.py`: AndroidWorld runner wrapper.
- `raw/scripts/`: emulator, AVD, sharded eval, and summarization scripts.
- `raw/task_sets/`: smoke, general-core, and general-focus task lists.
- `raw/runs/`: selected text logs and summaries, not heavy pkl/video artifacts.
- `raw/install_logs/`: install logs from SDK, pip, and AndroidWorld setup.
- `androidworld_git_state.txt`: AndroidWorld upstream commit and dirty files.
- `python_freeze.txt`: Python package freeze from the Vultr eval venv.
- `android_sdk_avd_inventory.txt`: installed SDK/AVD inventory.

Large artifacts are intentionally not committed to GitHub. Video and run
artifacts were uploaded separately to Hugging Face:

`https://huggingface.co/datasets/KMK040412/fast-dvlm-guiowl-androidworld-artifacts`

## Original Host Shape

- OS: Ubuntu 22.04 x64
- CPU: 24 cores / 48 threads
- RAM: 256 GB
- Storage: SSD/NVMe layout with `/data2` used for Android SDK, AVDs, and eval
- KVM: available and used by Android Emulator with `-accel on`

## Important Architecture

The evaluation host only ran Android emulators and AndroidWorld. The model
policy servers were on the RunPod GPU host.

On the Vultr eval host, ports `127.0.0.1:8123-8130` were SSH remote-forwarded
ports that reached eight RunPod GUI-Owl/Fast-dVLM HTTP servers. The AndroidWorld
runner talked only to these local forwarded ports.

Current bd32 dVLM server health looked like:

```json
{"ok": true, "model": "/workspace/dvlm_ckpts/ckpt_bard_bd32", "decode": "dvlm"}
```

## Key Evaluation Choice

Coordinate mode must be normalized:

```bash
GUIOWL_COORD_MODE=normalized
```

The earlier absolute-coordinate mode produced misleading failures. With
normalized coordinates, base GUI-Owl AR succeeded on smoke tasks, which confirmed
that the benchmark harness was able to register task-level success.

## Directory Layout To Recreate

```text
/data2/androidworld_eval
/data2/android-sdk
/data2/android-avd
/data2/android-emulator-logs
```

The backed-up scripts assume those paths. If a future host uses different paths,
update `raw/env.sh` and the scripts under `raw/scripts/`.

