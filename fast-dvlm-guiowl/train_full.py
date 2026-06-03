#!/usr/bin/env python
"""Full AITW-General distillation — streaming shards + PREFETCH dataloader, H100-optimised.

Throughput fixes vs the naive loop:
  * PREFETCH: CPU sample building (image decode + chat-template + processor) runs in
    background threads and feeds a bounded queue, overlapping with GPU training (the GPU was
    ~50% idle waiting on the CPU processor). The GPU vision forward + train stay in the main
    thread (single CUDA context). ~1.8x in practice.
  * Quality: forward.py now uses true vision M-RoPE + DeepStack (use_mrope/use_deepstack).

    HF_TOKEN=hf_xxx python train_full.py --config configs/h100.yaml \
        --model /workspace/models/GUI-Owl-1.5-2B-Instruct \
        --data-dir /workspace/data --out /workspace/ckpt --shard-end 256
"""
import argparse, math, os, queue, sys, threading, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fast_dvlm.data import iter_episodes, build_episode_sample, download_shard
from fast_dvlm.forward import dual_stream_loss


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mPLUG/GUI-Owl-1.5-2B-Instruct")
    p.add_argument("--data-dir", default="/workspace/data")
    p.add_argument("--out", default="/workspace/ckpt")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--shard-start", type=int, default=0)
    p.add_argument("--shard-end", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    # optimisation levers
    p.add_argument("--attn", choices=["sdpa", "flex"], default="flex")
    p.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true"); p.set_defaults(grad_ckpt=True)
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--freeze-vision", dest="freeze_vision", action="store_true"); p.set_defaults(freeze_vision=True)
    p.add_argument("--optim", choices=["adamw", "adamw_fused", "adamw_8bit"], default="adamw_fused")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--max-ep-steps", type=int, default=48)
    p.add_argument("--ctx-cap", type=int, default=16384)
    p.add_argument("--no-mrope", dest="mrope", action="store_false"); p.set_defaults(mrope=True)
    p.add_argument("--no-deepstack", dest="deepstack", action="store_false"); p.set_defaults(deepstack=True)
    p.add_argument("--kd-weight", type=float, default=0.0, help="teacher-KD KL weight (>0 enables distillation)")
    p.add_argument("--kd-temp", type=float, default=1.0)
    p.add_argument("--teacher", default="mPLUG/GUI-Owl-1.5-2B-Instruct", help="frozen AR teacher for KD")
    p.add_argument("--prefetch", type=int, default=12, help="bounded queue depth")
    p.add_argument("--producers", type=int, default=4, help="CPU sample-builder threads")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--anneal-steps", type=int, default=2000)
    p.add_argument("--bd-set", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after this many optimizer steps (0 = no limit)")
    p.add_argument("--early-stop-loss", type=float, default=0.0,
                   help="stop when recent train loss <= threshold (0 disables)")
    p.add_argument("--early-stop-window", type=int, default=50,
                   help="optimizer-step window for --early-stop-loss")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--no-final-save", action="store_true",
                   help="skip final model save, useful for throughput benchmarks")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def apply_yaml(a):
    if not a.config:
        return a
    import yaml
    cfg = yaml.safe_load(open(a.config)) or {}
    explicit = {x.split("=")[0].lstrip("-").replace("-", "_") for x in sys.argv[1:] if x.startswith("--")}
    keep = {"attn", "grad_ckpt", "freeze_vision", "optim", "grad_accum", "max_pixels",
            "max_ep_steps", "ctx_cap", "lr", "warmup", "bd_set"}
    for k, v in cfg.items():
        if k in keep and k not in explicit:
            setattr(a, k, v)
    return a


def shard_name(split, i):
    N = 256 if split == "train" else 32
    return f"standard/{split}-{i:05d}-of-{N:05d}.parquet"


def main():
    a = apply_yaml(parse())
    dev = "cuda"
    model_dtype = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.data_dir, exist_ok=True)

    if a.attn == "flex":
        from fast_dvlm.flex_attn import register_flex
        register_flex()
    attn_impl = "flex_block" if a.attn == "flex" else "sdpa"

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {a.model} attn={attn_impl} mrope={a.mrope} deepstack={a.deepstack} "
          f"prefetch={a.prefetch}/{a.producers}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=model_dtype, attn_implementation=attn_impl, low_cpu_mem_usage=True).to(dev)
    model.config.use_cache = False
    if a.freeze_vision:
        for prm in model.model.visual.parameters():
            prm.requires_grad_(False)
    (model.gradient_checkpointing_enable if a.grad_ckpt else model.gradient_checkpointing_disable)()
    model.train()
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)

    teacher = None
    if a.kd_weight > 0:
        teacher = AutoModelForImageTextToText.from_pretrained(
            a.teacher, dtype=model_dtype, attn_implementation="sdpa", low_cpu_mem_usage=True).to(dev).eval()
        for tp in teacher.parameters():
            tp.requires_grad_(False)
        print(f"[kd] frozen AR teacher loaded ({a.teacher}) weight={a.kd_weight} temp={a.kd_temp}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if a.optim == "adamw_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=a.lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                fused=(a.optim == "adamw_fused"))
    print(f"[opt] {a.optim} trainable {sum(p.numel() for p in trainable)/1e9:.2f}B "
          f"| weights VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # ----- producer/consumer prefetch -----
    q = queue.Queue(maxsize=a.prefetch)
    stop = threading.Event()
    gen_lock = threading.Lock()

    def episode_gen():
        for _ in range(a.epochs):
            for si in range(a.shard_start, a.shard_end):
                path = os.path.join(a.data_dir, shard_name(a.split, si))
                if not os.path.exists(path):
                    try:
                        path = download_shard(shard_name(a.split, si), a.data_dir)
                    except Exception as e:
                        print(f"[shard {si}] dl failed: {e}", flush=True); continue
                for er in iter_episodes(path, min_steps=2, max_steps=a.max_ep_steps):
                    yield er, si
    gen = episode_gen()

    def next_item():
        with gen_lock:
            try:
                return next(gen)
            except StopIteration:
                return None

    def producer():
        while not stop.is_set():
            item = next_item()
            if item is None:
                break
            er, si = item
            try:  # CPU-only build (no vision forward) -> overlaps with GPU training
                s = build_episode_sample(er, proc, model=None, cache_vision=False,
                                         device="cpu", max_pixels=a.max_pixels, ctx_cap=a.ctx_cap)
            except Exception:
                continue
            if s is None:
                continue
            s["_shard"] = si
            q.put(s)
        q.put(None)  # one sentinel per producer

    threads = [threading.Thread(target=producer, daemon=True) for _ in range(a.producers)]
    for t in threads:
        t.start()

    gstep, tok, micro, fin = 0, 0, 0, 0
    t0 = time.time(); losses = []
    stop_reason = None
    opt.zero_grad(set_to_none=True)
    while fin < a.producers:
        s = q.get()
        if s is None:
            fin += 1; continue
        u = min(gstep / max(a.anneal_steps, 1), 1.0)
        bd = a.bd_set[min(int(math.sqrt(u) * len(a.bd_set)), len(a.bd_set) - 1)]
        lr = a.lr * min((gstep + 1) / max(a.warmup, 1), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            loss, info = dual_stream_loss(model, s, bd, attn=a.attn,
                                          use_mrope=a.mrope, use_deepstack=a.deepstack,
                                          teacher=teacher, kd_weight=a.kd_weight, kd_temp=a.kd_temp)
            (loss / a.grad_accum).backward()
        except torch.cuda.OutOfMemoryError:
            print(f"[skip] OOM len={len(s['input_ids'])}", flush=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); micro = 0; continue
        tok += info["total"]; micro += 1
        if micro >= a.grad_accum:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); micro = 0
            gstep += 1; losses.append(loss.item())
            if gstep % a.log_every == 0:
                el = time.time() - t0
                kd_s = f" kd={info['kd']:.3f}" if 'kd' in info else ""
                print(f"[s{s['_shard']} step {gstep}] bd={bd} loss={loss.item():.3f} "
                      f"ce_c={info['ce_clean']:.3f}{kd_s} lr={lr:.1e} "
                      f"| {gstep/el:.2f} it/s {tok/el:.0f} tok/s | VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                      flush=True)
            if gstep % a.save_every == 0:
                model.save_pretrained(a.out); proc.save_pretrained(a.out)
                print(f"[ckpt] step {gstep} -> {a.out}", flush=True)
            if a.max_steps > 0 and gstep >= a.max_steps:
                stop_reason = f"max_steps={a.max_steps}"
                print(f"[stop] {stop_reason}", flush=True)
                stop.set()
                break
            if a.early_stop_loss > 0 and len(losses) >= a.early_stop_window:
                recent = sum(losses[-a.early_stop_window:]) / a.early_stop_window
                if recent <= a.early_stop_loss:
                    stop_reason = (f"rolling_loss={recent:.4f}<={a.early_stop_loss:.4f} "
                                   f"window={a.early_stop_window}")
                    print(f"[stop] {stop_reason}", flush=True)
                    stop.set()
                    break
    stop.set()
    if a.no_final_save:
        save_msg = f"not_saved {a.out}"
    else:
        model.save_pretrained(a.out); proc.save_pretrained(a.out)
        save_msg = f"saved {a.out}"
    if losses:
        suffix = f" | stop {stop_reason}" if stop_reason else ""
        print(f"[final] gstep={gstep} | loss {losses[0]:.3f}->{losses[-1]:.3f} | {save_msg}{suffix}", flush=True)
    else:
        print(f"[final] gstep=0 | no optimizer steps | {save_msg}", flush=True)
    print("DONE_FULL", flush=True)


if __name__ == "__main__":
    main()
