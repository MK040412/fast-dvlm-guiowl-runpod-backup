
from __future__ import annotations
import json, re


def parse_mobile_use(text: str):
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None, False
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None, False
    if isinstance(obj, dict) and obj.get("name") == "mobile_use" and isinstance(obj.get("arguments"), dict):
        return obj, True
    return None, False


def repair_mobile_use(text: str):
    obj, ok = parse_mobile_use(text)
    if ok:
        obj["repaired"] = False
        return obj, False
    if "mobile_use" not in text:
        return None, False
    am = re.search(r'"action"\s*:\s*"([^"\\]+)"', text)
    if not am:
        return None, False
    args = {"action": am.group(1)}
    coords = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if coords:
        args["coordinate"] = [int(float(coords[0][0])), int(float(coords[0][1]))]
    if len(coords) > 1:
        args["coordinate2"] = [int(float(coords[1][0])), int(float(coords[1][1]))]
    return {"name": "mobile_use", "arguments": args, "repaired": True}, True
