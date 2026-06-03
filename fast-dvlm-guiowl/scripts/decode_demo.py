#!/usr/bin/env python
"""Compare AR vs dVLM decoding and report GUI-Owl tool-call quality."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fast_dvlm.data import (  # noqa: E402
    MOBILE_USE_TOOL,
    SYSTEM_PROMPT,
    decode_image,
    iter_episodes,
    native_action,
)
from fast_dvlm.decode import ar_generate, dvlm_generate  # noqa: E402


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def first_json_object(text: str) -> str | None:
    match = TOOL_RE.search(text)
    if match:
        return match.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_mobile_use(raw: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "has_tool_call": bool(TOOL_RE.search(raw)),
        "valid_json": False,
        "valid_mobile_use": False,
        "parse_error": None,
    }
    obj_text = first_json_object(raw)
    if not obj_text:
        meta["parse_error"] = "no_json_object"
        return None, meta
    try:
        obj = json.loads(obj_text)
        meta["valid_json"] = True
    except Exception as exc:
        meta["parse_error"] = f"json_error:{type(exc).__name__}:{exc}"
        return None, meta

    if obj.get("name") == "mobile_use" and isinstance(obj.get("arguments"), dict):
        meta["valid_mobile_use"] = True
        return obj["arguments"], meta
    if isinstance(obj.get("action"), str):
        meta["valid_mobile_use"] = True
        return obj, meta
    meta["parse_error"] = "not_mobile_use"
    return None, meta


def coord_l2(pred: Any, target: Any) -> float | None:
    if not isinstance(pred, (list, tuple)) or not isinstance(target, (list, tuple)):
        return None
    if len(pred) < 2 or len(target) < 2:
        return None
    try:
        return math.hypot(float(pred[0]) - float(target[0]), float(pred[1]) - float(target[1]))
    except Exception:
        return None


def compare_prediction(raw: str, target_raw: str) -> dict[str, Any]:
    target_args, _ = parse_mobile_use(target_raw)
    pred_args, meta = parse_mobile_use(raw)
    out = dict(meta)
    out.update(
        {
            "strict_tool_call": bool(meta["has_tool_call"] and meta["valid_mobile_use"]),
            "action_match": False,
            "needs_coord": False,
            "coord_l2": None,
            "coord2_l2": None,
            "ground_ok_100": False,
        }
    )
    if not pred_args or not target_args:
        return out
    out["action_match"] = pred_args.get("action") == target_args.get("action")
    action = target_args.get("action")
    needs_coord = action in {"click", "long_press", "swipe"}
    out["needs_coord"] = needs_coord
    if needs_coord:
        out["coord_l2"] = coord_l2(pred_args.get("coordinate"), target_args.get("coordinate"))
        if action == "swipe":
            out["coord2_l2"] = coord_l2(pred_args.get("coordinate2"), target_args.get("coordinate2"))
        coord_ok = out["coord_l2"] is not None and out["coord_l2"] <= 100.0
        coord2_ok = action != "swipe" or (
            out["coord2_l2"] is not None and out["coord2_l2"] <= 100.0
        )
        out["ground_ok_100"] = bool(out["action_match"] and coord_ok and coord2_ok)
    else:
        out["ground_ok_100"] = bool(out["action_match"])
    return out


def yn(value: Any) -> str:
    return "Y" if value else "N"


def metric_line(m: dict[str, Any]) -> str:
    coord = "-" if m["coord_l2"] is None else f"{m['coord_l2']:.1f}"
    coord2 = "-" if m["coord2_l2"] is None else f"{m['coord2_l2']:.1f}"
    return (
        f"tool={yn(m['has_tool_call'])} json={yn(m['valid_json'])} "
        f"mobile={yn(m['valid_mobile_use'])} strict={yn(m['strict_tool_call'])} "
        f"act={yn(m['action_match'])} c1={coord} c2={coord2} "
        f"ground100={yn(m['ground_ok_100'])}"
    )


def summarize(name: str, items: list[dict[str, Any]]) -> None:
    n = len(items)
    if not n:
        print(f"{name}: no samples")
        return

    def rate(key: str) -> str:
        return f"{sum(bool(x[key]) for x in items)}/{n}"

    coord_vals = [x["coord_l2"] for x in items if x["coord_l2"] is not None]
    coord_text = "none"
    if coord_vals:
        coord_text = (
            f"mean={statistics.fmean(coord_vals):.1f} "
            f"median={statistics.median(coord_vals):.1f}"
        )
    print(
        f"{name}: tool_call {rate('has_tool_call')} | json {rate('valid_json')} | "
        f"mobile_use {rate('valid_mobile_use')} | strict {rate('strict_tool_call')} | "
        f"action {rate('action_match')} | ground@100 {rate('ground_ok_100')} | "
        f"coord_l2 {coord_text}"
    )


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--gen-len", type=int, default=32)
ap.add_argument("--tau", type=float, default=0.9)
ap.add_argument("--max-pixels", type=int, default=100352)
args = ap.parse_args()

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

model = AutoModelForImageTextToText.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
).to("cuda").eval()
proc = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

ar_tok = ar_ms = dv_tok = dv_nfe = dv_ms = 0
ar_metrics: list[dict[str, Any]] = []
dv_metrics: list[dict[str, Any]] = []
done = 0
for er in iter_episodes(args.data, min_steps=2, max_steps=9):
    img = decode_image(bytes(er.iloc[0]["image"]))
    if img is None:
        continue
    goal = er.iloc[0]["goal_info"]
    target = native_action(er.iloc[0])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Goal: {goal}"},
            ],
        },
    ]
    txt = proc.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=[MOBILE_USE_TOOL],
    )
    ii, vv = process_vision_info(messages) if process_vision_info else ([img], None)
    inp = proc(text=[txt], images=ii, videos=vv, return_tensors="pt").to("cuda")
    a_txt, a_n, a_ms = ar_generate(model, proc, inp, args.gen_len)
    d_txt, d_n, d_nfe, d_ms = dvlm_generate(model, proc, inp, args.gen_len, args.tau)
    ar_m = compare_prediction(a_txt, target)
    dv_m = compare_prediction(d_txt, target)
    print(f"target: {target}")
    print(f"  AR  : {a_txt!r:40s} NFE={a_n:2d} {a_ms:6.0f}ms | {metric_line(ar_m)}")
    print(
        f"  dVLM: {d_txt!r:40s} NFE={d_nfe:2d} "
        f"tok/NFE={d_n / max(d_nfe, 1):.2f} {d_ms:6.0f}ms | {metric_line(dv_m)}",
        flush=True,
    )
    ar_tok += a_n
    ar_ms += a_ms
    dv_tok += d_n
    dv_nfe += d_nfe
    dv_ms += d_ms
    ar_metrics.append(ar_m)
    dv_metrics.append(dv_m)
    done += 1
    if done >= args.n:
        break

if done == 0:
    raise SystemExit("no decodable held-out samples")

print(
    f"\nAVG: AR {ar_ms / done:.0f}ms (NFE {ar_tok / done:.1f}) | "
    f"dVLM {dv_ms / done:.0f}ms (NFE {dv_nfe / done:.1f}) | "
    f"tokens/NFE={dv_tok / max(dv_nfe, 1):.2f}x  wall {ar_ms / dv_ms:.2f}x"
)
summarize("AR", ar_metrics)
summarize("dVLM", dv_metrics)
