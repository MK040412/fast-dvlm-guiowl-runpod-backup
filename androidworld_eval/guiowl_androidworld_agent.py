"""AndroidWorld agent wrapper for GUI-Owl/Fast-dVLM policy server."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

import numpy as np
from PIL import Image
import requests

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ACTIONS = ("click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate")
BUTTONS = ("Home", "Back", "Enter", "Menu")


def _first_json_object(text: str) -> str | None:
    match = TOOL_RE.search(text)
    if match:
        return match.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
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
    for match in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", raw):
        x = int(round(float(match.group(1))))
        y = int(round(float(match.group(2))))
        pairs.append([max(0, min(1000, x)), max(0, min(1000, y))])
    return pairs


def _extract_button(raw: str) -> str | None:
    low = raw.lower()
    for button in BUTTONS:
        if button.lower() in low:
            return button
    return None


def _extract_text(raw: str) -> str:
    match = re.search(r'"text"\s*:\s*"([^"]{0,100})"', raw)
    return match.group(1) if match else ""


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


def parse_mobile_use(raw: str, repair: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "raw": raw,
        "has_tool_call": bool(TOOL_RE.search(raw)),
        "valid_json": False,
        "valid_mobile_use": False,
        "repaired": False,
        "parse_error": None,
    }
    obj_text = _first_json_object(raw)
    if not obj_text:
        meta["parse_error"] = "no_json_object"
    else:
        try:
            obj = json.loads(obj_text)
            meta["valid_json"] = True
        except Exception as exc:
            meta["parse_error"] = f"json_error:{type(exc).__name__}:{exc}"
        else:
            if obj.get("name") == "mobile_use" and isinstance(obj.get("arguments"), dict):
                args = obj["arguments"]
                meta["valid_mobile_use"] = True
                meta["tool_call"] = obj
                return args, meta

            if isinstance(obj.get("action"), str):
                meta["valid_mobile_use"] = True
                meta["tool_call"] = {"name": "mobile_use", "arguments": obj}
                return obj, meta

            meta["parse_error"] = "not_mobile_use"
            meta["tool_call"] = obj

    if repair:
        fixed = repair_mobile_use(raw)
        if fixed is not None:
            meta["repaired"] = True
            meta["valid_mobile_use"] = True
            meta["tool_call"] = {"name": "mobile_use", "arguments": fixed}
            return fixed, meta
    return None, meta


def _coord_to_pixel(coord: Any, screen_size: tuple[int, int]) -> tuple[int, int] | None:
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    width, height = screen_size
    x = max(0, min(1000, float(coord[0])))
    y = max(0, min(1000, float(coord[1])))
    return int(round(x / 1000.0 * width)), int(round(y / 1000.0 * height))


def _swipe_direction(args: dict[str, Any]) -> str:
    start = args.get("coordinate")
    end = args.get("coordinate2")
    if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"
    return str(args.get("direction") or "up")


def to_android_action(
    args: dict[str, Any] | None,
    screen_size: tuple[int, int],
) -> tuple[json_action.JSONAction, dict[str, Any]]:
    meta: dict[str, Any] = {"android_action_valid": False, "action_error": None}
    if not args:
        meta["action_error"] = "no_arguments"
        return json_action.JSONAction(action_type="wait"), meta

    action = str(args.get("action", "")).lower()
    try:
        if action == "click":
            xy = _coord_to_pixel(args.get("coordinate"), screen_size)
            if xy is None:
                raise ValueError("click_missing_coordinate")
            out = json_action.JSONAction(action_type="click", x=xy[0], y=xy[1])
        elif action == "long_press":
            xy = _coord_to_pixel(args.get("coordinate"), screen_size)
            if xy is None:
                raise ValueError("long_press_missing_coordinate")
            out = json_action.JSONAction(action_type="long_press", x=xy[0], y=xy[1])
        elif action == "swipe":
            out = json_action.JSONAction(
                action_type="swipe",
                direction=_swipe_direction(args),
            )
        elif action == "type":
            out = json_action.JSONAction(
                action_type="input_text",
                text=str(args.get("text", "")),
                clear_text=True,
            )
        elif action == "system_button":
            button = str(args.get("button", "")).lower()
            if button == "home":
                out = json_action.JSONAction(action_type="navigate_home")
            elif button == "back":
                out = json_action.JSONAction(action_type="navigate_back")
            elif button == "enter":
                out = json_action.JSONAction(action_type="keyboard_enter")
            else:
                out = json_action.JSONAction(action_type="wait")
        elif action == "open":
            out = json_action.JSONAction(
                action_type="open_app",
                app_name=str(args.get("text") or args.get("app_name") or ""),
            )
        elif action == "wait":
            out = json_action.JSONAction(action_type="wait")
        elif action == "terminate":
            status = str(args.get("status", "success")).lower()
            out = json_action.JSONAction(
                action_type="status",
                goal_status="complete" if status == "success" else "infeasible",
            )
        else:
            raise ValueError(f"unsupported_action:{action}")
        meta["android_action_valid"] = True
        meta["android_action"] = out.as_dict()
        return out, meta
    except Exception as exc:
        meta["action_error"] = f"{type(exc).__name__}:{exc}"
        return json_action.JSONAction(action_type="wait"), meta


def _screenshot_b64(pixels: np.ndarray) -> str:
    image = Image.fromarray(pixels.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class GuiOwlAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        server_url: str = "http://127.0.0.1:8123",
        timeout: float = 120.0,
    ):
        super().__init__(env, name="guiowl_dvlm")
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.history: list[str] = []

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        state = self.get_post_transition_state()
        screen_size = self.env.logical_screen_size
        request = {
            "goal": goal,
            "screen_width": screen_size[0],
            "screen_height": screen_size[1],
            "history": self.history[-8:],
            "screenshot_b64": _screenshot_b64(state.pixels),
        }
        t0 = time.time()
        response = requests.post(
            f"{self.server_url}/predict",
            json=request,
            timeout=self.timeout,
        )
        response.raise_for_status()
        pred = response.json()
        raw = pred.get("raw", "")
        repair = os.environ.get("GUIOWL_REPAIR", "1") != "0"
        args, parse_meta = parse_mobile_use(raw, repair=repair)
        action, action_meta = to_android_action(args, screen_size)
        done = action.action_type == "status" and action.goal_status == "complete"

        step_data = {
            "goal": goal,
            "screen_size": screen_size,
            "server_prediction": pred,
            "parsed_arguments": args,
            "structural": parse_meta,
            "android_action": action.as_dict(),
            "action_meta": action_meta,
            "latency_total_ms": int((time.time() - t0) * 1000),
        }

        self.history.append(json.dumps(step_data["android_action"], sort_keys=True))
        if action.action_type != "status":
            self.env.execute_action(action)
        return base_agent.AgentInteractionResult(done, step_data)


if __name__ == "__main__":
    examples = [
        '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[695,793]}}\n</tool_call>',
        '<tool_call>\n{"name":"mobile_use","arguments":{"action":"system_button","button":"Home"}}\n</tool_call>',
        "broken",
    ]
    for raw in examples:
        args, meta = parse_mobile_use(raw)
        action, ameta = to_android_action(args, (540, 1080))
        print(raw)
        print(args)
        print(meta)
        print(action, ameta)
