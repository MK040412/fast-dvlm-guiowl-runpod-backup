#!/usr/bin/env python
"""Fast-dVLM training entrypoint (GUI-Owl-1.5-2B / Qwen3-VL -> block-diffusion VLM).

Single-device, throughput-oriented. All optimisation levers are flags; the H100 preset
turns them all on. See README.md for the rationale + measured speedups.

Examples
--------
# L40S, verified default (sdpa, grad-ckpt, frozen vision):
python train.py --data ./data/standard/train-00000-of-00256.parquet --steps 300 --overfit

# H100 SXM, max throughput:
python train.py --config configs/h100.yaml --data './data/standard/train-*.parquet'
"""
import argparse, glob, math, time, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fast_dvlm.data import load_samples
from fast_dvlm.forward import dual_stream_loss


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mPLUG/GUI-Owl-1.5-2B-Instruct")
    p.add_argument("--data", required=True, help="parquet path or glob")
    p.add_argument("--config", default=None, help="yaml preset (overridden by explicit flags)")
    # optimisation levers
    p.add_argument("--attn", choices=["sdpa", "flex"], default="sdpa",
                   help="flex = FlexAttention BlockMask (H100 fast path, ~2-2.5x attn)")
    p.add_argument("--compile", action="store_true", help="torch.compile the LM")
    p.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.set_defaults(grad_ckpt=True)
    p.add_argument("--freeze-vision", dest="freeze_vision", action="store_true")
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")
    p.set_defaults(freeze_vision=True)
    p.add_argument("--optim", choices=["adamw", "adamw_fused", "adamw_8bit"], default="adamw_fused")
    p.add_argument("--grad-accum", type=int, default=1, help="effective batch via accumulation")
    p.add_argument("--max-pixels", type=int, default=200704, help="vision tokens lever (lower = faster)")
    # data / training
    p.add_argument("--n-samples", type=int, default=0, help="0 = all episodes in the shard(s)")
    p.add_argument("--max-ep-steps", type=int, default=64)
    p.add_argument("--ctx-cap", type=int, default=16384)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--bd-set", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--overfit", action="store_true", help="cycle a few samples (fast memorisation demo)")
    p.add_argument("--save", default=None)
    return p.parse_args()


def apply_yaml(args):
    if not args.config:
        return args
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    explicit = {a.split("=")[0].lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    for k, v in cfg.items():
        if k not in explicit:
            setattr(args, k, v)
    return args


def main():
    args = apply_yaml(parse())
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.attn == "flex":
        from fast_dvlm.flex_attn import register_flex
        register_flex()
    attn_impl = "flex_block" if args.attn == "flex" else "sdpa"

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {args.model} (attn={attn_impl})", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attn_impl, low_cpu_mem_usage=True).to(dev)
    model.config.use_cache = False
    if args.freeze_vision:
        for prm in model.model.visual.parameters():
            prm.requires_grad_(False)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.train()
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    if args.compile:
        model.model.language_model = torch.compile(model.model.language_model)

    files = sorted(glob.glob(args.data)) or [args.data]
    samples = []
    for fp in files:
        samples += load_samples(fp, proc, model, n=(args.n_samples or None),
                                max_steps=args.max_ep_steps, max_pixels=args.max_pixels,
                                device=dev, cache_vision=args.freeze_vision, ctx_cap=args.ctx_cap)
        if args.n_samples and len(samples) >= args.n_samples:
            break
    print(f"[data] {len(samples)} episodes | lens {[len(s['input_ids']) for s in samples][:12]}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.0, fused=(args.optim == "adamw_fused"))
    print(f"[opt] {args.optim} | trainable {sum(p.numel() for p in trainable)/1e9:.2f}B", flush=True)

    torch.cuda.reset_peak_memory_stats()
    tstart, tok_seen, losses = time.time(), 0, []
    for step in range(args.steps):
        u = step / max(args.steps - 1, 1)
        bd = args.bd_set[min(int(math.sqrt(u) * len(args.bd_set)), len(args.bd_set) - 1)]
        opt.zero_grad(set_to_none=True)
        agg = 0.0
        for micro in range(args.grad_accum):
            idx = (step * args.grad_accum + micro)
            s = samples[idx % len(samples)] if args.overfit else samples[idx % len(samples)]
            loss, info = dual_stream_loss(model, s, bd, attn=args.attn)
            (loss / args.grad_accum).backward()
            agg += loss.item() / args.grad_accum
            tok_seen += info["total"]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(agg)
        if step % 20 == 0 or step == args.steps - 1:
            el = time.time() - tstart
            print(f"[{step:4d}] bd={bd:2d} loss={agg:6.3f} | {step/max(el,1e-9):.2f} it/s "
                  f"| {tok_seen/max(el,1e-9):7.0f} tok/s | peakVRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                  flush=True)
    print(f"[done] loss {losses[0]:.3f} -> {losses[-1]:.3f} (min {min(losses):.3f}) | "
          f"{args.steps/(time.time()-tstart):.2f} it/s | peakVRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB",
          flush=True)
    if args.save:
        model.save_pretrained(args.save); proc.save_pretrained(args.save)
        print(f"[save] {args.save}", flush=True)


if __name__ == "__main__":
    main()
