
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steering", required=True)
    p.add_argument("--data", default="/workspace/data/standard")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--bd", type=int, default=32)
    p.add_argument("--repair", type=str, default="true")
    p.add_argument("--compare-base", type=str, default="true")
    p.add_argument("--out", default="/workspace/paper_logs/bd32_steering_sft_eval.json")
    args = p.parse_args()
    # Full decode wiring is intentionally kept separate from training. This placeholder records
    # the strict/repaired split expected by paper logs and fails loudly if treated as a benchmark.
    report = {
        "status": "not_run_decode_mvp",
        "reason": "training adapter path is implemented; generation-time steering decode must be wired to fast_dvlm.decode before reporting benchmark numbers",
        "strict_json_count": 0,
        "strict_json_rate": None,
        "mobile_use_valid_count": 0,
        "mobile_use_valid_rate": None,
        "repaired_mobile_use_valid_count": 0,
        "repaired_mobile_use_valid_rate": None,
        "repaired_count": 0,
        "average_decode_ms_base": None,
        "average_decode_ms_steering": None,
        "steering_overhead_ms": None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(2)

if __name__ == "__main__":
    main()
