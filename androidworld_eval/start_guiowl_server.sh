#!/bin/bash
set -euo pipefail

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export TOKENIZERS_PARALLELISM=false
export GUIOWL_MODEL=${GUIOWL_MODEL:-/opt/dvlm_ckpts/ckpt_bard_bd8}
export GUIOWL_DECODE=${GUIOWL_DECODE:-dvlm}
export GUIOWL_SERVER_PORT=${GUIOWL_SERVER_PORT:-8123}
export GUIOWL_MAX_PIXELS=${GUIOWL_MAX_PIXELS:-100352}
export GUIOWL_GEN_LEN=${GUIOWL_GEN_LEN:-64}
export GUIOWL_TAU=${GUIOWL_TAU:-0.9}

cd /opt/androidworld_eval
exec /workspace/venv/bin/python /opt/androidworld_eval/guiowl_action_server.py
