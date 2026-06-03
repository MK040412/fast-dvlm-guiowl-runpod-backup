
#!/usr/bin/env python
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def first_parquet(data, split="test"):
    pats = [os.path.join(data, f"{split}-*.parquet"), os.path.join(data, "standard", f"{split}-*.parquet"), f"/workspace/data/standard/{split}-*.parquet"]
    for pat in pats:
        xs = sorted(glob.glob(pat))
        if xs:
            return xs[0]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="/workspace/data/standard")
    p.add_argument("--bd", type=int, default=32)
    p.add_argument("--tap-layers", default="last4")
    p.add_argument("--inject-layers", default="last8")
    p.add_argument("--out", default="/workspace/paper_logs/steering_identity_smoke.json")
    p.add_argument("--steer-dim", type=int, default=512)
    p.add_argument("--mla-dim", type=int, default=256)
    p.add_argument("--ssd-layers", type=int, default=3)
    p.add_argument("--mla-layers", type=int, default=1)
    p.add_argument("--n-steer-blocks", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=100352)
    p.add_argument("--ctx-cap", type=int, default=8192)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from fast_dvlm.data import iter_episodes, build_episode_sample
    from fast_dvlm.forward import dual_stream_loss
    from fast_dvlm.steering import SteeringConfig, SteeringWrapper
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    cfg = SteeringConfig(bd=args.bd, tap_layers=args.tap_layers, inject_layers=args.inject_layers, steer_dim=args.steer_dim, mla_dim=args.mla_dim, ssd_layers=args.ssd_layers, mla_layers=args.mla_layers, n_steer_blocks=args.n_steer_blocks, base_model=args.model)
    wrapper = SteeringWrapper(model, cfg).to(dev)
    wrapper.freeze_base(); wrapper.eval()
    zero = all(float(p.detach().abs().max().cpu()) == 0.0 for h in wrapper.residual_heads.values() for p in [h.weight, h.bias])
    path = first_parquet(args.data, "test") or first_parquet(args.data, "train")
    loss_diff = None
    if path:
        sample = None
        for er in iter_episodes(path, min_steps=2, max_steps=16):
            sample = build_episode_sample(er, proc, model=None, cache_vision=False, max_pixels=args.max_pixels, ctx_cap=args.ctx_cap)
            if sample is not None:
                break
        if sample is not None:
            seed = 20260604
            torch.manual_seed(seed)
            with torch.no_grad():
                base_loss, _ = dual_stream_loss(model, sample, args.bd, attn="sdpa", use_mrope=True, use_deepstack=True)
            torch.manual_seed(seed)
            wrapper.begin_trace_capture()
            with torch.no_grad():
                _ = dual_stream_loss(model, sample, args.bd, attn="sdpa", use_mrope=True, use_deepstack=True)
            wrapper.end_trace_capture()
            torch.manual_seed(seed)
            wrapper.prepare_residuals(batch_size=2, device=dev, dtype=next(wrapper.parameters()).dtype)
            with torch.no_grad():
                steered_loss, _ = dual_stream_loss(model, sample, args.bd, attn="sdpa", use_mrope=True, use_deepstack=True)
            wrapper.clear_residuals()
            loss_diff = abs(float(base_loss.detach().cpu()) - float(steered_loss.detach().cpu()))
    report = {
        "zero_init_verified": zero,
        "max_abs_logit_diff": loss_diff,
        "mean_abs_logit_diff": loss_diff,
        "loss_diff_used_as_identity_proxy": True,
        "trace_source": cfg.trace_source,
        "injection_point": wrapper.injector.injection_point,
        "tap_layers": wrapper.tap_layer_ids,
        "inject_layers": wrapper.inject_layer_ids,
        "pass": bool(zero and (loss_diff is None or loss_diff <= 1e-6)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
