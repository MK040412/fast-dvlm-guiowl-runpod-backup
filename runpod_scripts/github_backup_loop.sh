#!/bin/bash
set -euo pipefail

INTERVAL=${GITHUB_BACKUP_INTERVAL:-900}
LOG=${GITHUB_BACKUP_LOG:-/workspace/github_backup_loop.log}

while true; do
  date -u '+[github-loop] %Y-%m-%dT%H:%M:%SZ start' >> "$LOG"
  /workspace/androidworld_eval/venv/bin/python /workspace/github_backup.py >> "$LOG" 2>&1 || true
  date -u '+[github-loop] %Y-%m-%dT%H:%M:%SZ done' >> "$LOG"
  sleep "$INTERVAL"
done
