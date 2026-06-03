#!/usr/bin/env python
from __future__ import annotations
import argparse, gzip, json, pickle
from pathlib import Path
from statistics import mean
SUCCESS_KEYS = ("is_successful", "is_success", "successful")
TASK_KEYS = ("task_template", "task_name", "name")
GOAL_KEYS = ("goal", "task_prompt")
RUNTIME_KEYS = ("run_time", "runtime", "total_runtime_s")
LENGTH_KEYS = ("episode_length", "length")
EXC_KEYS = ("exception_info", "exception")

def first(ep, keys, default=None):
    for k in keys:
        if k in ep:
            return ep[k]
    return default

def as_float(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.lower() in {"nan", "none", ""}: return None
        v = float(x)
        if v != v: return None
        return v
    except Exception:
        return None

def load_episodes(root: Path):
    episodes = []
    for path in sorted(root.rglob("*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as f:
                obj = pickle.load(f)
            items = obj if isinstance(obj, list) else [obj]
            for ep in items:
                if isinstance(ep, dict):
                    ep = dict(ep)
                    ep["_file"] = str(path)
                    episodes.append(ep)
        except Exception as e:
            episodes.append({"_file": str(path), "exception_info": f"load_error: {type(e).__name__}: {e}"})
    return episodes

def summarize(root: Path):
    episodes = load_episodes(root)
    by_task, valid_scores, failures = {}, [], []
    for ep in episodes:
        task = first(ep, TASK_KEYS, "unknown")
        score = as_float(first(ep, SUCCESS_KEYS))
        exc = first(ep, EXC_KEYS)
        runtime = as_float(first(ep, RUNTIME_KEYS))
        length = as_float(first(ep, LENGTH_KEYS))
        item = by_task.setdefault(task, {"n": 0, "success_sum": 0.0, "success_rate": None, "runtime_mean_s": None, "episode_length_mean": None, "exceptions": 0, "episodes": []})
        item["n"] += 1
        if score is not None:
            item["success_sum"] += score
            valid_scores.append(score)
        if exc not in (None, "", "None"):
            item["exceptions"] += 1
            failures.append({"task": task, "exception": str(exc)[:500], "file": ep.get("_file")})
        item["episodes"].append({"goal": first(ep, GOAL_KEYS), "is_successful": score, "runtime_s": runtime, "episode_length": length, "exception": None if exc in (None, "", "None") else str(exc)[:500], "file": ep.get("_file")})
    for item in by_task.values():
        scores = [e["is_successful"] for e in item["episodes"] if e["is_successful"] is not None]
        runtimes = [e["runtime_s"] for e in item["episodes"] if e["runtime_s"] is not None]
        lengths = [e["episode_length"] for e in item["episodes"] if e["episode_length"] is not None]
        item["success_rate"] = sum(scores) / len(scores) if scores else None
        item["runtime_mean_s"] = mean(runtimes) if runtimes else None
        item["episode_length_mean"] = mean(lengths) if lengths else None
    return {"run_root": str(root), "checkpoint_files": [str(p) for p in sorted(root.rglob("*.pkl.gz"))], "episode_count": len(episodes), "scored_episode_count": len(valid_scores), "success_count": sum(valid_scores) if valid_scores else None, "success_rate": sum(valid_scores) / len(valid_scores) if valid_scores else None, "tasks": by_task, "failures": failures}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = summarize(Path(args.root))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"root": result["run_root"], "episode_count": result["episode_count"], "success_rate": result["success_rate"], "tasks": {k: {"n": v["n"], "success_rate": v["success_rate"]} for k, v in result["tasks"].items()}}, indent=2))
if __name__ == "__main__": main()
