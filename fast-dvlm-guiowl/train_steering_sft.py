
#!/usr/bin/env python
from __future__ import annotations
import argparse, glob, json, math, os, queue, sys, threading, time
from pathlib import Path
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fast_dvlm.data import iter_episodes, build_episode_sample
from fast_dvlm.forward import dual_stream_loss
from fast_dvlm.steering import SteeringConfig, SteeringWrapper
from fast_dvlm.steering.config import str2bool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="/workspace/data/standard")
    p.add_argument("--split", default="train")
    p.add_argument("--bd", type=int, default=32)
    p.add_argument("--mode", default="teacher_forced_trace_sft")
    p.add_argument("--tap-layers", default="last4")
    p.add_argument("--inject-layers", default="last8")
    p.add_argument("--steer-dim", type=int, default=512)
    p.add_argument("--mla-dim", type=int, default=256)
    p.add_argument("--ssd-layers", type=int, default=3)
    p.add_argument("--mla-layers", type=int, default=1)
    p.add_argument("--n-steer-blocks", type=int, default=2)
    p.add_argument("--gate-center", type=float, default=0.35)
    p.add_argument("--gate-width", type=float, default=0.15)
    p.add_argument("--learn-gate", type=str2bool, default=False)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optim", default="adamw_fused")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--attn", choices=["sdpa", "flex"], default="sdpa")
    p.add_argument("--max-pixels", type=int, default=100352)
    p.add_argument("--ctx-cap", type=int, default=8192)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--freeze-base", type=str2bool, default=True)
    p.add_argument("--train-steering-only", type=str2bool, default=True)
    p.add_argument("--struct-weight", type=float, default=1.5)
    p.add_argument("--coord-token-weight", type=float, default=2.0)
    p.add_argument("--lambda-action", type=float, default=0.1)
    p.add_argument("--lambda-coord", type=float, default=0.05)
    p.add_argument("--lambda-res", type=float, default=1e-4)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--steps-per-block-train", type=int, default=2)
    p.add_argument("--disable-aux-heads", type=str2bool, default=False)
    p.add_argument("--trace-source", default="auto")
    p.add_argument("--action-token-slice", default="last_bd")
    p.add_argument("--save-adapter-only", type=str2bool, default=True)
    p.add_argument("--resume-steering", default=None)
    p.add_argument("--producers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=8)
    p.add_argument("--max-ep-steps", type=int, default=48)
    p.add_argument("--no-mrope", dest="mrope", action="store_false"); p.set_defaults(mrope=True)
    p.add_argument("--no-deepstack", dest="deepstack", action="store_false"); p.set_defaults(deepstack=True)
    return p.parse_args()


def parquet_files(data, split):
    pats = [os.path.join(data, f"{split}-*.parquet"), os.path.join(data, "standard", f"{split}-*.parquet")]
    out = []
    for pat in pats:
        out += sorted(glob.glob(pat))
    return sorted(set(out))


def base_grad_nonzero_count(model):
    n = 0
    for p in model.parameters():
        if p.grad is not None and bool((p.grad.detach().abs() > 0).any().item()):
            n += 1
    return n


def save_train_log(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def main():
    a = parse_args()
    if a.mode != "teacher_forced_trace_sft":
        raise ValueError("this first implementation supports teacher_forced_trace_sft")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.jsonl").write_text("")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    if a.attn == "flex":
        from fast_dvlm.flex_attn import register_flex
        register_flex()
    attn_impl = "flex_block" if a.attn == "flex" else "sdpa"
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] base={a.model} dtype={a.dtype} attn={attn_impl}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=model_dtype, attn_implementation=attn_impl, low_cpu_mem_usage=True).to(dev)
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)
    cfg = SteeringConfig(
        bd=a.bd, tap_layers=a.tap_layers, inject_layers=a.inject_layers, steer_dim=a.steer_dim,
        mla_dim=a.mla_dim, ssd_layers=a.ssd_layers, mla_layers=a.mla_layers, n_steer_blocks=a.n_steer_blocks,
        gate_center=a.gate_center, gate_width=a.gate_width, learn_gate=a.learn_gate,
        lambda_action=a.lambda_action, lambda_coord=a.lambda_coord, lambda_res=a.lambda_res,
        struct_weight=a.struct_weight, coord_token_weight=a.coord_token_weight,
        train_steps_per_block=a.steps_per_block_train, action_token_slice=a.action_token_slice,
        base_model=a.model,
    )
    wrapper = SteeringWrapper(model, cfg).to(dev)
    wrapper.freeze_base(); wrapper.train()
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    trainable_names = [n for n, p in wrapper.named_parameters() if p.requires_grad]
    if a.train_steering_only and any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("base parameter is trainable")
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay, fused=(a.optim == "adamw_fused" and dev == "cuda"))
    print(f"[steering] params={sum(p.numel() for p in trainable):,} tap={wrapper.tap_layer_ids} inject={wrapper.inject_layer_ids} trace={cfg.trace_source} injection={wrapper.injector.injection_point}", flush=True)
    print(f"[base] frozen={all(not p.requires_grad for p in model.parameters())} base_params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    cfg.save_json(out / "steering_config.json")
    (out / "trainable_names.json").write_text(json.dumps(trainable_names, indent=2))

    files = parquet_files(a.data, a.split)
    if not files:
        raise FileNotFoundError(f"no parquet files for split={a.split} under {a.data}")
    print(f"[data] {len(files)} parquet files; first={files[0]}", flush=True)

    q = queue.Queue(maxsize=a.prefetch)
    stop = threading.Event()
    lock = threading.Lock()
    produced = {"n": 0}

    def episode_iter():
        for ep in range(a.epochs):
            for path in files:
                for er in iter_episodes(path, min_steps=2, max_steps=a.max_ep_steps):
                    yield ep, path, er
    gen = episode_iter()

    def next_item():
        with lock:
            if a.max_train_samples and produced["n"] >= a.max_train_samples:
                return None
            try:
                item = next(gen)
                produced["n"] += 1
                return item
            except StopIteration:
                return None

    def producer():
        while not stop.is_set():
            item = next_item()
            if item is None:
                break
            ep, path, er = item
            try:
                s = build_episode_sample(er, proc, model=None, cache_vision=False, max_pixels=a.max_pixels, ctx_cap=a.ctx_cap)
            except Exception as e:
                continue
            if s is None:
                continue
            s["_epoch"] = ep; s["_path"] = path
            q.put(s)
        q.put(None)

    threads = [threading.Thread(target=producer, daemon=True) for _ in range(max(a.producers, 1))]
    for t in threads:
        t.start()

    opt.zero_grad(set_to_none=True)
    gstep = 0; micro = 0; seen = 0; finished = 0; tok = 0
    t0 = time.time(); last_loss = None
    while finished < len(threads):
        s = q.get()
        if s is None:
            finished += 1
            continue
        seen += 1
        seed = 1000003 + seen
        try:
            wrapper.clear_residuals()
            torch.manual_seed(seed)
            wrapper.begin_trace_capture()
            with torch.no_grad():
                _base_loss, _base_info = dual_stream_loss(model, s, a.bd, attn=a.attn, use_mrope=a.mrope, use_deepstack=a.deepstack)
            wrapper.end_trace_capture()

            torch.manual_seed(seed)
            wrapper.prepare_residuals(batch_size=2, device=dev, dtype=next(wrapper.parameters()).dtype, progress=a.gate_center)
            loss, info = dual_stream_loss(model, s, a.bd, attn=a.attn, use_mrope=a.mrope, use_deepstack=a.deepstack)
            res_loss = wrapper.last_res_loss if wrapper.last_res_loss is not None else torch.tensor(0.0, device=dev)
            total_loss = loss + a.lambda_res * res_loss
            (total_loss / a.grad_accum).backward()
            wrapper.clear_residuals()
        except torch.cuda.OutOfMemoryError:
            print(f"[skip] OOM len={len(s['input_ids'])}", flush=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); micro = 0; wrapper.clear_residuals(); continue
        micro += 1; tok += int(info.get("total", 0)); last_loss = float(total_loss.detach().cpu())
        if micro >= a.grad_accum:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
            bg = base_grad_nonzero_count(model)
            if bg != 0:
                raise RuntimeError(f"base_grad_nonzero_count={bg}")
            opt.step(); opt.zero_grad(set_to_none=True); micro = 0; gstep += 1
            elapsed = max(time.time() - t0, 1e-6)
            row = {
                "step": gstep, "seen_samples": seen, "loss": last_loss, "ce_loss": float(loss.detach().cpu()),
                "ce_noisy": info.get("ce_noisy"), "ce_clean": info.get("ce_clean"),
                "action_loss": None, "coord_loss": None, "res_loss": float(res_loss.detach().cpu()),
                "residual_norm_mean": wrapper.last_residual_norm_mean, "residual_norm_max": wrapper.last_residual_norm_max,
                "gate_mean": wrapper.last_gate, "trace_frames_mean": len(wrapper.trace_bank), "trace_source": cfg.trace_source,
                "injection_point": wrapper.injector.injection_point, "grad_norm_steering": grad_norm,
                "base_grad_nonzero_count": bg, "lr": opt.param_groups[0]["lr"],
                "tokens_per_sec": tok / elapsed, "examples_per_sec": seen / elapsed,
                "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            }
            save_train_log(out / "train_log.jsonl", row)
            if gstep % a.log_every == 0:
                print(f"[step {gstep}] loss={row['loss']:.4f} ce={row['ce_loss']:.4f} res={row['res_loss']:.6f} trace={row['trace_frames_mean']} grad={grad_norm:.3f} {row['examples_per_sec']:.2f} ex/s {row['tokens_per_sec']:.0f} tok/s vram={row['vram_gb']:.1f}GB", flush=True)
            if gstep % a.save_every == 0:
                wrapper.save_adapter(out, vars(a))
                print(f"[ckpt] adapter step={gstep} -> {out}", flush=True)
            if a.max_steps and gstep >= a.max_steps:
                stop.set(); break
    if micro > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
        bg = base_grad_nonzero_count(model)
        if bg != 0:
            raise RuntimeError(f"base_grad_nonzero_count={bg}")
        opt.step(); opt.zero_grad(set_to_none=True); gstep += 1
        elapsed = max(time.time() - t0, 1e-6)
        row = {"step": gstep, "seen_samples": seen, "loss": last_loss, "grad_norm_steering": grad_norm, "base_grad_nonzero_count": bg, "tokens_per_sec": tok / elapsed, "examples_per_sec": seen / elapsed, "trace_source": cfg.trace_source, "injection_point": wrapper.injector.injection_point}
        save_train_log(out / "train_log.jsonl", row)
    wrapper.save_adapter(out, vars(a))
    print(f"[done] steps={gstep} samples={seen} out={out}", flush=True)

if __name__ == "__main__":
    main()
