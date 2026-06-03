#!/usr/bin/env python
"""Write compact paper-oriented metrics for one BARD stage."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


FINAL_RE = re.compile(r"\[final\].*gstep=(\d+).*loss\s+([0-9.]+)->([0-9.]+)")
STEP_RE = re.compile(
    r"\[s(?P<shard>\d+)\s+step\s+(?P<step>\d+)\].*bd=(?P<bd>\d+)\s+"
    r"loss=(?P<loss>[0-9.]+).*ce_c=(?P<ce>[0-9.]+).*lr=(?P<lr>[0-9.e+-]+).*"
    r"\|\s+(?P<itps>[0-9.]+)\s+it/s\s+(?P<tokps>[0-9.]+)\s+tok/s\s+\| VRAM\s+(?P<vram>[0-9.]+)GB"
)
COMPARE_RE = re.compile(
    r"^(?P<name>[^:]+): avg_ms=(?P<avg_ms>[0-9.]+) avg_nfe=(?P<avg_nfe>[0-9.]+) \| "
    r"tool_call (?P<tool>\d+/\d+) \| json (?P<json>\d+/\d+) \| "
    r"mobile_use (?P<mobile>\d+/\d+) \| strict (?P<strict>\d+/\d+) \| "
    r"repaired (?P<repaired>\d+/\d+) \| action (?P<action>\d+/\d+) \| "
    r"ground@100 (?P<ground>\d+/\d+) \| coord_l2 (?P<coord>.*)$"
)
DECODE_AVG_RE = re.compile(
    r"AVG: AR (?P<ar_ms>[0-9.]+)ms \(NFE (?P<ar_nfe>[0-9.]+)\) \| "
    r"dVLM (?P<dvlm_ms>[0-9.]+)ms \(NFE (?P<dvlm_nfe>[0-9.]+)\) \| "
    r"tokens/NFE=(?P<tok_nfe>[0-9.]+)x\s+wall (?P<wall>[0-9.]+)x"
)


def read(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    return p.read_text(errors="replace") if p.exists() else ""


def parse_ratio(text: str) -> dict[str, int | float]:
    a, b = text.split("/", 1)
    num = int(a)
    den = int(b)
    return {"num": num, "den": den, "rate": (num / den if den else 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--bd", type=int, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-log", default="")
    ap.add_argument("--decode-log", default="")
    ap.add_argument("--compare-log", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    model_file = ckpt / "model.safetensors"

    train_txt = read(args.train_log)
    decode_txt = read(args.decode_log)
    compare_txt = read(args.compare_log)

    final = None
    for m in FINAL_RE.finditer(train_txt):
        final = {
            "gstep": int(m.group(1)),
            "loss_start": float(m.group(2)),
            "loss_end": float(m.group(3)),
        }

    last_step = None
    step_count = 0
    for m in STEP_RE.finditer(train_txt):
        step_count += 1
        last_step = {
            "shard": int(m.group("shard")),
            "step": int(m.group("step")),
            "bd": int(m.group("bd")),
            "loss": float(m.group("loss")),
            "ce_clean": float(m.group("ce")),
            "lr": m.group("lr"),
            "it_per_sec": float(m.group("itps")),
            "tok_per_sec": float(m.group("tokps")),
            "vram_gb": float(m.group("vram")),
        }

    compare = {}
    for line in compare_txt.splitlines():
        m = COMPARE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        compare[d["name"]] = {
            "avg_ms": float(d["avg_ms"]),
            "avg_nfe": float(d["avg_nfe"]),
            "tool_call": parse_ratio(d["tool"]),
            "json": parse_ratio(d["json"]),
            "mobile_use": parse_ratio(d["mobile"]),
            "strict": parse_ratio(d["strict"]),
            "repaired": parse_ratio(d["repaired"]),
            "action": parse_ratio(d["action"]),
            "ground_at_100": parse_ratio(d["ground"]),
            "coord_l2": d["coord"],
        }

    decode_avg = None
    for m in DECODE_AVG_RE.finditer(decode_txt):
        decode_avg = {k: float(v) for k, v in m.groupdict().items()}

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "bd": args.bd,
        "epoch": args.epoch,
        "source_checkpoint": args.source,
        "checkpoint": str(ckpt),
        "checkpoint_exists": model_file.exists(),
        "checkpoint_model_safetensors_bytes": model_file.stat().st_size if model_file.exists() else 0,
        "train": {
            "final": final,
            "last_logged_step": last_step,
            "logged_step_count": step_count,
        },
        "decode_demo": {
            "avg": decode_avg,
            "log": args.decode_log,
        },
        "grounding_compare": {
            "metrics": compare,
            "log": args.compare_log,
        },
        "logs": {
            "train": args.train_log,
            "decode": args.decode_log,
            "compare": args.compare_log,
        },
        "config": {
            "dtype": os.environ.get("DTYPE", "bf16"),
            "max_pixels": int(os.environ.get("MAX_PIXELS", "100352")),
            "grad_accum": int(os.environ.get("GRAD_ACCUM", "8")),
            "lr": os.environ.get("LR", "3e-5"),
            "prefetch": int(os.environ.get("PREFETCH", "32")),
            "producers": int(os.environ.get("PRODUCERS", "8")),
            "max_ep_steps": int(os.environ.get("MAX_EP_STEPS", "24")),
            "ctx_cap": int(os.environ.get("CTX_CAP", "8192")),
            "save_every": int(os.environ.get("SAVE_EVERY", "100")),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with open("/workspace/paper_logs/stage_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(f"[paper] wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
