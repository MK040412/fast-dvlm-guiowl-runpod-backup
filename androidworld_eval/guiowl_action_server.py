#!/usr/bin/env python
"""Small HTTP policy server for GUI-Owl/Fast-dVLM AndroidWorld eval.

Run this with the training venv because it needs torch/transformers and the
fast_dvlm package. The AndroidWorld agent calls /predict with a screenshot and
goal, then converts the returned raw tool_call into AndroidWorld JSONAction.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/workspace/fast-dvlm-guiowl/src")
from fast_dvlm.data import MOBILE_USE_TOOL, SYSTEM_PROMPT  # noqa: E402
from fast_dvlm.decode import ar_generate, dvlm_generate  # noqa: E402

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None


MODEL_PATH = os.environ.get("GUIOWL_MODEL", "/workspace/ckpt_bard_bd2")
MODE = os.environ.get("GUIOWL_DECODE", "dvlm")
PORT = int(os.environ.get("GUIOWL_SERVER_PORT", "8123"))
MAX_PIXELS = int(os.environ.get("GUIOWL_MAX_PIXELS", "100352"))
GEN_LEN = int(os.environ.get("GUIOWL_GEN_LEN", "64"))
TAU = float(os.environ.get("GUIOWL_TAU", "0.9"))


def _load_model():
    dtype = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, max_pixels=MAX_PIXELS)
    return model, processor


MODEL, PROCESSOR = _load_model()


def predict(payload: dict) -> dict:
    img_bytes = base64.b64decode(payload["screenshot_b64"])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    goal = payload.get("goal", "")
    history = payload.get("history") or []

    history_text = ""
    if history:
        clipped = history[-8:]
        history_text = "\nPrevious actions:\n" + "\n".join(clipped)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Goal: {goal}{history_text}"},
            ],
        },
    ]
    prompt = PROCESSOR.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=[MOBILE_USE_TOOL],
    )
    images, videos = (
        process_vision_info(messages) if process_vision_info else ([image], None)
    )
    inputs = PROCESSOR(
        text=[prompt],
        images=images,
        videos=videos,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        if MODE == "ar":
            raw, n_tok, ms = ar_generate(MODEL, PROCESSOR, inputs, GEN_LEN)
            nfe = n_tok
        else:
            raw, n_tok, nfe, ms = dvlm_generate(MODEL, PROCESSOR, inputs, GEN_LEN, TAU)

    return {
        "raw": raw,
        "decode": MODE,
        "tokens": n_tok,
        "nfe": nfe,
        "latency_ms": ms,
        "model": MODEL_PATH,
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, data: dict) -> None:
        blob = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "model": MODEL_PATH, "decode": MODE})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._json(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            self._json(200, predict(payload))
        except Exception as exc:  # pragma: no cover
            self._json(500, {"error": type(exc).__name__, "message": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


if __name__ == "__main__":
    print(
        f"[guiowl_server] model={MODEL_PATH} decode={MODE} port={PORT} "
        f"max_pixels={MAX_PIXELS}",
        flush=True,
    )
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
