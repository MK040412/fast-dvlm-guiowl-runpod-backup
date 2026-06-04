#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
source /data2/androidworld_eval/venv/bin/activate
TASK_SET=${TASK_SET:-general_core}
SUITE_FAMILY=${SUITE_FAMILY:-android_world}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30}
LANES_PER_MODEL=${LANES_PER_MODEL:-4}
BASE_COORD_MODE=${BASE_COORD_MODE:-absolute}
CURRENT_COORD_MODE=${CURRENT_COORD_MODE:-absolute}
OUT_ROOT=${OUT_ROOT:-/data2/androidworld_eval/runs/$(date +%Y%m%d_%H%M%S)_${TASK_SET}_8lane}
mkdir -p "$OUT_ROOT/shards"
get_tasks() {
  if [ "$TASK_SET" = "standard_full" ]; then
    python - <<'PY'
from android_world import registry
r=registry.TaskRegistry().get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY)
print(','.join(sorted(r.keys())))
PY
  elif [ -f "/data2/androidworld_eval/task_sets/${TASK_SET}.txt" ]; then
    tr -d '\n ' < "/data2/androidworld_eval/task_sets/${TASK_SET}.txt"
  else
    echo "$TASK_SET"
  fi
}
TASKS_CSV=$(get_tasks)
python - "$TASKS_CSV" "$LANES_PER_MODEL" "$OUT_ROOT/shards" <<'PY'
import json, pathlib, sys
tasks=[t for t in sys.argv[1].split(',') if t]
lanes=int(sys.argv[2]); out=pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
shards=[]
for i in range(lanes):
    shard=tasks[i::lanes]
    (out/f'shard_{i}.txt').write_text(','.join(shard), encoding='utf-8')
    shards.append({'shard': i, 'n_tasks': len(shard), 'tasks': shard})
(out/'shards.json').write_text(json.dumps({'total_tasks': len(tasks), 'lanes': lanes, 'shards': shards}, indent=2), encoding='utf-8')
print(json.dumps({'total_tasks': len(tasks), 'lanes': lanes, 'shard_sizes': [s['n_tasks'] for s in shards]}))
PY
cat > "$OUT_ROOT/plan.txt" <<EOF
TASK_SET=$TASK_SET
SUITE_FAMILY=$SUITE_FAMILY
N_TASK_COMBINATIONS=$N_TASK_COMBINATIONS
TASK_RANDOM_SEED=$TASK_RANDOM_SEED
LANES_PER_MODEL=$LANES_PER_MODEL
BASE_PORTS=8123,8125,8127,8129
CURRENT_PORTS=8124,8126,8128,8130
BASE_CONSOLES=5554,5556,5558,5560
CURRENT_CONSOLES=5562,5564,5566,5568
EOF
base_ports=(8123 8125 8127 8129)
cur_ports=(8124 8126 8128 8130)
base_consoles=(5554 5556 5558 5560)
cur_consoles=(5562 5564 5566 5568)
base_grpcs=(8554 8555 8556 8557)
cur_grpcs=(8558 8559 8560 8561)
pids=()
run_lane() {
  local model=$1 shard=$2 url=$3 console=$4 grpc=$5
  local shard_file="$OUT_ROOT/shards/shard_${shard}.txt"
  local tasks
  tasks=$(cat "$shard_file")
  local name="${model}_${TASK_SET}_shard${shard}"
  echo "[lane] model=$model shard=$shard console=$console grpc=$grpc url=$url tasks=$(python - <<PY
print(len([t for t in '''$tasks'''.split(',') if t]))
PY
)"
  local coord_mode="$BASE_COORD_MODE"
  if [ "$model" = "current" ]; then coord_mode="$CURRENT_COORD_MODE"; fi
  NAME="$name" SERVER_URL="$url" CONSOLE_PORT="$console" GRPC_PORT="$grpc" GUIOWL_COORD_MODE="$coord_mode" \
  TASKS="$tasks" SUITE_FAMILY="$SUITE_FAMILY" N_TASK_COMBINATIONS="$N_TASK_COMBINATIONS" TASK_RANDOM_SEED="$TASK_RANDOM_SEED" \
  OUT_ROOT="$OUT_ROOT" /data2/androidworld_eval/scripts/run_one_eval.sh \
  >"$OUT_ROOT/${name}.runner.log" 2>&1 &
  pids+=("$!")
}
for i in $(seq 0 $((LANES_PER_MODEL-1))); do
  run_lane base "$i" "http://127.0.0.1:${base_ports[$i]}" "${base_consoles[$i]}" "${base_grpcs[$i]}"
  run_lane current "$i" "http://127.0.0.1:${cur_ports[$i]}" "${cur_consoles[$i]}" "${cur_grpcs[$i]}"
done
status=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then status=1; fi
done
/data2/androidworld_eval/scripts/summarize_androidworld_run.py "$OUT_ROOT" | tee "$OUT_ROOT/summary.stdout"
echo "DONE $OUT_ROOT status=$status"
exit "$status"
