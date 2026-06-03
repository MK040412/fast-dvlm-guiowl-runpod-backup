#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data2/androidworld_eval/runs/large_8lane_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ROOT"
echo "ROOT=$ROOT" | tee "$ROOT/plan.txt"
echo "TASK_SETS=general_core,standard_full" | tee -a "$ROOT/plan.txt"
echo "LANES_PER_MODEL=4" | tee -a "$ROOT/plan.txt"
echo "N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}" | tee -a "$ROOT/plan.txt"
echo "TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30}" | tee -a "$ROOT/plan.txt"
for task_set in general_core standard_full; do
  echo "[suite] start $task_set $(date -Is)" | tee -a "$ROOT/plan.txt"
  OUT_ROOT="$ROOT/${task_set}" TASK_SET="$task_set" LANES_PER_MODEL=4 N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1} TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30} \
    /data2/androidworld_eval/scripts/run_eval_sharded_8lane.sh \
    > "$ROOT/${task_set}.nohup.log" 2>&1
  echo "[suite] done $task_set $(date -Is)" | tee -a "$ROOT/plan.txt"
done
python - <<PY
import json, pathlib
root=pathlib.Path("$ROOT")
combined={"root": str(root), "suites": {}}
for name in ["general_core", "standard_full"]:
    p=root/name/"summary.json"
    if p.exists():
        combined["suites"][name]=json.loads(p.read_text()).get("aggregate", {})
(root/"combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(json.dumps(combined, indent=2))
PY
