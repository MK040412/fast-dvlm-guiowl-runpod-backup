#!/usr/bin/env python
"""Compare GUI-Owl source model and a trained dVLM checkpoint on AITW grounding.

The report separates strict JSON/tool-call validity from conservative structural
repair, because malformed dVLM text should not be silently counted as valid.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
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
ACTIONS = ("click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate")
BUTTONS = ("Home", "Back", "Enter", "Menu")


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


def _extract_action(raw: str) -> str | None:
    low = raw.lower()
    for action in ACTIONS:
        if re.search(rf'["\']?action["\']?\s*:{{0,2}}\s*["\']?{re.escape(action)}', low):
            return action
    for action in ACTIONS:
        if action in low:
            return action
    return None


def _extract_coord_pairs(raw: str) -> list[list[int]]:
    pairs: list[list[int]] = []
    for m in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", raw):
        x = int(round(float(m.group(1))))
        y = int(round(float(m.group(2))))
        pairs.append([max(0, min(1000, x)), max(0, min(1000, y))])
    return pairs


def _extract_button(raw: str) -> str | None:
    low = raw.lower()
    for button in BUTTONS:
        if button.lower() in low:
            return button
    return None


def _extract_text(raw: str) -> str:
    m = re.search(r'"text"\s*:\s*"([^"]{0,100})"', raw)
    return m.group(1) if m else ""


def repair_mobile_use(raw: str) -> dict[str, Any] | None:
    action = _extract_action(raw)
    if not action:
        return None
    args: dict[str, Any] = {"action": action}
    pairs = _extract_coord_pairs(raw)
    if action in {"click", "long_press"}:
        if not pairs:
            return None
        args["coordinate"] = pairs[0]
    elif action == "swipe":
        if len(pairs) < 2:
            return None
        args["coordinate"] = pairs[0]
        args["coordinate2"] = pairs[1]
    elif action == "system_button":
        button = _extract_button(raw)
        if not button:
            return None
        args["button"] = button
    elif action == "type":
        args["text"] = _extract_text(raw)
    elif action == "terminate":
        args["status"] = "failure" if "failure" in raw.lower() else "success"
    return args


def parse_mobile_use(raw: str, *, repair: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "raw": raw,
        "has_tool_call": bool(TOOL_RE.search(raw)),
        "valid_json": False,
        "valid_mobile_use": False,
        "repaired": False,
        "parse_error": None,
    }
    obj_text = first_json_object(raw)
    if obj_text:
        try:
            obj = json.loads(obj_text)
            meta["valid_json"] = True
            if obj.get("name") == "mobile_use" and isinstance(obj.get("arguments"), dict):
                meta["valid_mobile_use"] = True
                return obj["arguments"], meta
            if isinstance(obj.get("action"), str):
                meta["valid_mobile_use"] = True
                return obj, meta
            meta["parse_error"] = "not_mobile_use"
        except Exception as exc:
            meta["parse_error"] = f"json_error:{type(exc).__name__}:{exc}"
    else:
        meta["parse_error"] = "no_json_object"

    if repair:
        fixed = repair_mobile_use(raw)
        if fixed is not None:
            meta["repaired"] = True
            meta["valid_mobile_use"] = True
            return fixed, meta
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


def compare_prediction(raw: str, target_raw: str, *, repair: bool) -> dict[str, Any]:
    target_args, _ = parse_mobile_use(target_raw, repair=False)
    pred_args, meta = parse_mobile_use(raw, repair=repair)
    out = dict(meta)
    out.update(
        {
            "strict_tool_call": bool(meta["has_tool_call"] and meta["valid_json"] and meta["valid_mobile_use"]),
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
    out["needs_coord"] = action in {"click", "long_press", "swipe"}
    if out["needs_coord"]:
        out["coord_l2"] = coord_l2(pred_args.get("coordinate"), target_args.get("coordinate"))
        if action == "swipe":
            out["coord2_l2"] = coord_l2(pred_args.get("coordinate2"), target_args.get("coordinate2"))
        coord_ok = out["coord_l2"] is not None and out["coord_l2"] <= 100.0
        coord2_ok = action != "swipe" or (out["coord2_l2"] is not None and out["coord2_l2"] <= 100.0)
        out["ground_ok_100"] = bool(out["action_match"] and coord_ok and coord2_ok)
    else:
        out["ground_ok_100"] = bool(out["action_match"])
    return out


@dataclass
class ModelSpec:
    name: str
    path: str
    mode: str
    repair: bool


def summarize(name: str, rows: list[dict[str, Any]], total_ms: float, total_nfe: float) -> None:
    n = len(rows)
    if not n:
        print(f"{name}: no samples")
        return

    def rate(key: str) -> str:
        return f"{sum(bool(x[key]) for x in rows)}/{n}"

    coords = [x["coord_l2"] for x in rows if x["coord_l2"] is not None]
    coord_text = "none"
    if coords:
        coord_text = f"mean={statistics.fmean(coords):.1f} median={statistics.median(coords):.1f}"
    print(
        f"{name}: avg_ms={total_ms/n:.0f} avg_nfe={total_nfe/n:.1f} | "
        f"tool_call {rate('has_tool_call')} | json {rate('valid_json')} | "
        f"mobile_use {rate('valid_mobile_use')} | strict {rate('strict_tool_call')} | "
        f"repaired {rate('repaired')} | action {rate('action_match')} | "
        f"ground@100 {rate('ground_ok_100')} | coord_l2 {coord_text}"
    )


def evaluate(
    spec: ModelSpec,
    data: str,
    n: int,
    gen_len: int,
    tau: float,
    max_pixels: int,
    episode_max_steps: int,
) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"\n[load] {spec.name} mode={spec.mode} repair={spec.repair} path={spec.path}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        spec.path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(spec.path, max_pixels=max_pixels)
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        process_vision_info = None

    rows: list[dict[str, Any]] = []
    total_ms = 0.0
    total_nfe = 0.0
    done = 0
    for er in iter_episodes(data, min_steps=2, max_steps=episode_max_steps):
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
        txt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=[MOBILE_USE_TOOL])
        ii, vv = process_vision_info(messages) if process_vision_info else ([img], None)
        inp = proc(text=[txt], images=ii, videos=vv, return_tensors="pt").to("cuda")
        with torch.no_grad():
            if spec.mode == "ar":
                raw, n_tok, ms = ar_generate(model, proc, inp, gen_len)
                nfe = n_tok
            else:
                raw, n_tok, nfe, ms = dvlm_generate(model, proc, inp, gen_len, tau)
        m = compare_prediction(raw, target, repair=spec.repair)
        rows.append(m)
        total_ms += ms
        total_nfe += nfe
        done += 1
        if done <= 5:
            print(f"sample={done} target={target}")
            print(f"  raw={raw!r}")
            print(
                "  metrics="
                f"tool={m['has_tool_call']} json={m['valid_json']} mobile={m['valid_mobile_use']} "
                f"repaired={m['repaired']} action={m['action_match']} "
                f"coord_l2={m['coord_l2']} ground100={m['ground_ok_100']}"
            )
        if done >= n:
            break
    summarize(spec.name, rows, total_ms, total_nfe)
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()


def parse_spec(text: str) -> ModelSpec:
    # name=path:mode[:repair|strict]
    name, rest = text.split("=", 1)
    parts = rest.split(":")
    if len(parts) < 2:
        raise ValueError(f"bad spec {text!r}; expected name=path:mode[:repair]")
    path = parts[0]
    mode = parts[1]
    repair = len(parts) >= 3 and parts[2] == "repair"
    if mode not in {"ar", "dvlm"}:
        raise ValueError(f"bad mode {mode!r}")
    return ModelSpec(name, path, mode, repair)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/standard/test-00000-of-00032.parquet")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--max-pixels", type=int, default=100352)
    ap.add_argument("--episode-max-steps", type=int, default=24)
    ap.add_argument(
        "--spec",
        action="append",
        default=[],
        help="name=path:ar|dvlm[:repair]. Can be repeated.",
    )
    args = ap.parse_args()
    specs = args.spec or [
        "source_ar=/workspace/models/GUI-Owl-1.5-2B-Instruct:ar",
        "final_ar=/opt/dvlm_ckpts/ckpt_bard_bd8:ar",
        "final_dvlm_strict=/opt/dvlm_ckpts/ckpt_bard_bd8:dvlm",
        "final_dvlm_repair=/opt/dvlm_ckpts/ckpt_bard_bd8:dvlm:repair",
    ]
    for spec_text in specs:
        evaluate(
            parse_spec(spec_text),
            args.data,
            args.n,
            args.gen_len,
            args.tau,
            args.max_pixels,
            args.episode_max_steps,
        )


if __name__ == "__main__":
    main()
