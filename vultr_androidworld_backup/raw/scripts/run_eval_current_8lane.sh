#!/usr/bin/env bash
set -euo pipefail

source /data2/androidworld_eval/env.sh
source /data2/androidworld_eval/venv/bin/activate

TASK_SET=${TASK_SET:-general_core}
SUITE_FAMILY=${SUITE_FAMILY:-android_world}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30}
LANES=${LANES:-8}
CURRENT_COORD_MODE=${CURRENT_COORD_MODE:-normalized}
OUT_ROOT=${OUT_ROOT:-/data2/androidworld_eval/runs/$(date +%Y%m%d_%H%M%S)_${TASK_SET}_current_8lane}

mkdir -p "$OUT_ROOT/shards"

get_tasks() {
  if [ "$TASK_SET" = "standard_full" ]; then
    python - <<'PY'
from android_world import registry
r = registry.TaskRegistry().get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY)
print(",".join(sorted(r.keys())))
PY
  elif [ -f "/data2/androidworld_eval/task_sets/${TASK_SET}.txt" ]; then
    tr -d '\n ' < "/data2/androidworld_eval/task_sets/${TASK_SET}.txt"
  else
    echo "$TASK_SET"
  fi
}

TASKS_CSV=$(get_tasks)
python - "$TASKS_CSV" "$LANES" "$OUT_ROOT/shards" <<'PY'
import json
import pathlib
import sys

tasks = [t for t in sys.argv[1].split(",") if t]
lanes = int(sys.argv[2])
out = pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

shards = []
for i in range(lanes):
    shard = tasks[i::lanes]
    (out / f"shard_{i}.txt").write_text(",".join(shard), encoding="utf-8")
    shards.append({"shard": i, "n_tasks": len(shard), "tasks": shard})

(out / "shards.json").write_text(
    json.dumps({"total_tasks": len(tasks), "lanes": lanes, "shards": shards}, indent=2),
    encoding="utf-8",
)
print(json.dumps({"total_tasks": len(tasks), "lanes": lanes, "shard_sizes": [s["n_tasks"] for s in shards]}))
PY

cat > "$OUT_ROOT/plan.txt" <<PLAN
TASK_SET=$TASK_SET
SUITE_FAMILY=$SUITE_FAMILY
N_TASK_COMBINATIONS=$N_TASK_COMBINATIONS
TASK_RANDOM_SEED=$TASK_RANDOM_SEED
LANES=$LANES
MODEL=current_bd32_dvlm
COORD_MODE=$CURRENT_COORD_MODE
PORTS=8123,8124,8125,8126,8127,8128,8129,8130
CONSOLES=5554,5556,5558,5560,5562,5564,5566,5568
PLAN

ports=(8123 8124 8125 8126 8127 8128 8129 8130)
consoles=(5554 5556 5558 5560 5562 5564 5566 5568)
grpcs=(8554 8555 8556 8557 8558 8559 8560 8561)
pids=()

for i in $(seq 0 $((LANES - 1))); do
  tasks=$(cat "$OUT_ROOT/shards/shard_${i}.txt")
  name="current_${TASK_SET}_shard${i}"
  n_tasks=$(python - "$tasks" <<'PY'
import sys
print(len([t for t in sys.argv[1].split(",") if t]))
PY
)
  echo "[lane] model=current shard=$i console=${consoles[$i]} grpc=${grpcs[$i]} url=http://127.0.0.1:${ports[$i]} tasks=${n_tasks}"
  NAME="$name" SERVER_URL="http://127.0.0.1:${ports[$i]}" CONSOLE_PORT="${consoles[$i]}" GRPC_PORT="${grpcs[$i]}" \
  GUIOWL_COORD_MODE="$CURRENT_COORD_MODE" GUIOWL_RECORD_TRAJ="${GUIOWL_RECORD_TRAJ:-0}" \
  TASKS="$tasks" SUITE_FAMILY="$SUITE_FAMILY" N_TASK_COMBINATIONS="$N_TASK_COMBINATIONS" TASK_RANDOM_SEED="$TASK_RANDOM_SEED" \
  OUT_ROOT="$OUT_ROOT" /data2/androidworld_eval/scripts/run_one_eval.sh \
  >"$OUT_ROOT/${name}.runner.log" 2>&1 &
  pids+=("$!")
done

status=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    status=1
  fi
done

/data2/androidworld_eval/scripts/summarize_androidworld_run.py "$OUT_ROOT" | tee "$OUT_ROOT/summary.stdout"
echo "DONE $OUT_ROOT status=$status"
exit "$status"
