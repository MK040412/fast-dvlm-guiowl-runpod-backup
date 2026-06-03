"""AITW-General -> Qwen3-VL multi-turn SFT samples for Fast-dVLM training.

Key dataset facts (HF ``cjfcsjt/AITW_General``, ``standard`` split):
  * 256 train + 32 test parquet shards (~100MB each). Download one shard at a time.
  * ``image`` is RAW RGB bytes (NOT png), decoded by device_type:
        PIXEL_3 -> 540x1080 (1749600B), PIXEL_4 -> 540x1140 (1846800B)  [W=540]
        CUSTOM_DEVICE (904752B) uses other dims -> filtered out.
  * One row per step: ep_id, step_id, episode_length, goal_info, results_action_type,
    results_yx_touch/lift, results_type_action.

Each *episode* becomes ONE multi-turn sequence (whole trajectory, not truncated): the
Fast-dVLM multi-turn mask treats each step's action as its own response block/turn.
"""
from __future__ import annotations
import io, os, glob, json
import numpy as np
import torch
from PIL import Image

ATYPE = {3: "TYPE", 4: "DUAL_POINT", 5: "BACK", 6: "HOME", 7: "ENTER", 10: "COMPLETE", 11: "IMPOSSIBLE"}

# GUI-Owl native action space (verified by prompting the teacher: it outputs a Qwen
# tool_call with mobile_use + 0-1000 coordinates, and grounds correctly). Using the native
# format lets the teacher's existing grounding transfer instead of being relearned.
MOBILE_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": ("Use a touchscreen to interact with a mobile device. Coordinates are "
                        "(x, y) on a 0-1000 normalized scale; origin is the top-left."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["click", "long_press", "swipe", "type",
                                    "system_button", "open", "wait", "terminate"]},
                "coordinate": {"type": "array", "description": "(x, y) target, 0-1000."},
                "coordinate2": {"type": "array", "description": "(x, y) swipe end, 0-1000."},
                "text": {"type": "string"},
                "button": {"type": "string", "enum": ["Back", "Home", "Menu", "Enter"]},
                "status": {"type": "string", "enum": ["success", "failure"]},
            },
            "required": ["action"],
        },
    },
}
SYSTEM_PROMPT = ("You are a GUI agent operating an Android phone. Given the goal, the action "
                 "history and the current screenshot, output the next action by calling the "
                 "mobile_use function.")


def _xy(yx):  # AITW stores [y, x] in 0-1 -> native [x, y] in 0-1000
    return [int(round(float(yx[1]) * 1000)), int(round(float(yx[0]) * 1000))]


def native_action(r) -> str:
    """AITW row -> GUI-Owl native ``<tool_call>{...}</tool_call>`` string."""
    a = int(r["results_action_type"])
    if a == 4:
        t = np.asarray(r["results_yx_touch"], float); l = np.asarray(r["results_yx_lift"], float)
        if np.allclose(t, l, atol=1e-3):
            args = {"action": "click", "coordinate": _xy(t)}
        else:
            args = {"action": "swipe", "coordinate": _xy(t), "coordinate2": _xy(l)}
    elif a == 3:
        args = {"action": "type", "text": "".join(r["results_type_action"])[:100]}
    elif a == 5:
        args = {"action": "system_button", "button": "Back"}
    elif a == 6:
        args = {"action": "system_button", "button": "Home"}
    elif a == 7:
        args = {"action": "system_button", "button": "Enter"}
    elif a == 10:
        args = {"action": "terminate", "status": "success"}
    elif a == 11:
        args = {"action": "terminate", "status": "failure"}
    else:
        args = {"action": "wait", "time": 1}
    return "<tool_call>\n" + json.dumps({"name": "mobile_use", "arguments": args}) + "\n</tool_call>"


# backward-compat alias (decode_demo / older scripts)
format_action = native_action


def decode_image(raw: bytes, width: int = 540):
    """AITW raw-RGB decode. Returns None for non-``width`` devices (e.g. CUSTOM_DEVICE)."""
    arr = np.frombuffer(raw, np.uint8)
    if arr.size % (width * 3) != 0:
        return None
    h = arr.size // 3 // width
    return Image.fromarray(arr.reshape(h, width, 3), "RGB")


def download_shard(shard: str, out_dir: str, token: str | None = None) -> str:
    """Download one parquet shard (e.g. ``standard/train-00000-of-00256.parquet``)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download("cjfcsjt/AITW_General", shard, repo_type="dataset",
                           local_dir=out_dir, token=token or os.environ.get("HF_TOKEN"))


def iter_episodes(parquet_path: str, min_steps=2, max_steps=64):
    import pyarrow.parquet as pq
    df = pq.ParquetFile(parquet_path).read().to_pandas()
    for ep, _ in df.groupby("ep_id"):
        er = df[df["ep_id"] == ep].sort_values("step_id").reset_index(drop=True)
        if min_steps <= len(er) <= max_steps:
            yield er


def build_episode_sample(er, processor, *, max_pixels=200704, device="cuda",
                         model=None, cache_vision=True, ctx_cap=None, dtype=torch.bfloat16):
    """One episode -> dict(input_ids, labels, vemb?, pixel_values?, image_grid_thw?, meta...).

    If ``model`` is given and ``cache_vision``, the frozen vision embeds are precomputed and
    returned in ``vemb`` (so the training loop never re-runs the ViT). Otherwise raw
    ``pixel_values``/``image_grid_thw`` are returned for an in-loop vision forward.
    """
    tok = processor.tokenizer
    ASSIST = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    IM_END = tok.encode("<|im_end|>", add_special_tokens=False)[0]
    imgs = [decode_image(bytes(er.iloc[k]["image"])) for k in range(len(er))]
    if any(x is None for x in imgs):
        return None
    goal = er.iloc[0]["goal_info"]
    actions = [native_action(er.iloc[k]) for k in range(len(er))]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for k in range(len(er)):
        content = [{"type": "image", "image": imgs[k]}]
        if k == 0:
            content.append({"type": "text", "text": f"Goal: {goal}"})
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": actions[k]})  # native tool_call string
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                         tools=[MOBILE_USE_TOOL])
    try:
        from qwen_vl_utils import process_vision_info
        ii, vv = process_vision_info(msgs)
    except Exception:
        ii, vv = imgs, None
    b = processor(text=[text], images=ii, videos=vv, return_tensors="pt")
    ids = b["input_ids"][0]
    if ctx_cap and len(ids) > ctx_cap:
        return None
    # assistant-only labels
    labels = torch.full_like(ids, -100)
    T, seq, i = len(ASSIST), ids.tolist(), 0
    while i <= len(seq) - T:
        if seq[i:i + T] == ASSIST:
            s = e = i + T
            while e < len(seq) and seq[e] != IM_END:
                e += 1
            e = min(e + 1, len(seq))
            labels[s:e] = ids[s:e]
            i = e
        else:
            i += 1
    out = dict(input_ids=ids, labels=labels, n_steps=len(er), goal=goal,
               actions=actions, first_img=imgs[0])
    # mm_token_type_ids is emitted by the Qwen3-VL processor and is required for true
    # multimodal RoPE (M-RoPE). Keep it on CPU; the train loop moves it to device.
    mmt = b.get("mm_token_type_ids", None)
    out["mm_token_type_ids"] = mmt[0] if mmt is not None else None
    # Always keep pixel_values/image_grid_thw: the forward needs them for DeepStack features
    # (get_image_features returns pooler_output AND deepstack_features).
    out["pixel_values"] = b["pixel_values"]
    out["image_grid_thw"] = b["image_grid_thw"]
    if model is not None and cache_vision:
        with torch.no_grad():
            vo = model.model.get_image_features(b["pixel_values"].to(device), b["image_grid_thw"].to(device))
            out["vemb"] = torch.cat(list(vo.pooler_output), 0).to(dtype)
    return out


def load_samples(parquet_path, processor, model, n=None, **kw):
    samples = []
    for er in iter_episodes(parquet_path, min_steps=kw.pop("min_steps", 2),
                            max_steps=kw.pop("max_steps", 64)):
        s = build_episode_sample(er, processor, model=model, **kw)
        if s is not None:
            samples.append(s)
        if n and len(samples) >= n:
            break
    return samples
