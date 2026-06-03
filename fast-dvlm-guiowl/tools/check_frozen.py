
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def fingerprint(model):
    out = {}
    with __import__('torch').no_grad():
        for name, p in model.named_parameters():
            d = p.detach()
            out[name] = (float(d.float().sum().cpu()), float(d.float().abs().max().cpu()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="/workspace/paper_logs/steering_frozen_check.json")
    p.add_argument("--bd", type=int, default=32)
    p.add_argument("--tap-layers", default="last4")
    p.add_argument("--inject-layers", default="last8")
    p.add_argument("--steer-dim", type=int, default=512)
    p.add_argument("--mla-dim", type=int, default=256)
    p.add_argument("--ssd-layers", type=int, default=3)
    p.add_argument("--mla-layers", type=int, default=1)
    p.add_argument("--n-steer-blocks", type=int, default=2)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText
    from fast_dvlm.steering import SteeringConfig, SteeringWrapper
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    cfg = SteeringConfig(bd=args.bd, tap_layers=args.tap_layers, inject_layers=args.inject_layers, steer_dim=args.steer_dim, mla_dim=args.mla_dim, ssd_layers=args.ssd_layers, mla_layers=args.mla_layers, n_steer_blocks=args.n_steer_blocks, base_model=args.model)
    wrapper = SteeringWrapper(model, cfg)
    wrapper.freeze_base()
    wrapper.train()
    trainable = [("steering." + n, p) for n, p in wrapper.named_parameters() if p.requires_grad]
    base_param_count = sum(p.numel() for p in model.parameters())
    steering_param_count = sum(p.numel() for _, p in trainable)
    before = fingerprint(model)
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
    trace = torch.randn(1, 1, args.bd, wrapper.hidden_dim, device=next(wrapper.parameters()).device, dtype=next(wrapper.parameters()).dtype)
    wrapper.trace_bank.reset()
    from fast_dvlm.steering.trace_bank import TraceFrame
    wrapper.trace_bank.append(TraceFrame(step_index=0, hidden=trace[:, 0], layer_id=-1))
    wrapper.prepare_residuals(batch_size=1, device=trace.device, dtype=trace.dtype)
    loss = wrapper.last_res_loss
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    after = fingerprint(model)
    changed = []
    maxdiff = 0.0
    for k in before:
        if before[k] != after[k]:
            changed.append(k)
            maxdiff = max(maxdiff, abs(before[k][0] - after[k][0]), abs(before[k][1] - after[k][1]))
    report = {
        "base_param_count": base_param_count,
        "steering_param_count": steering_param_count,
        "trainable_names": [n for n, _ in trainable],
        "changed_base_params": changed,
        "max_abs_base_param_diff": maxdiff,
        "base_frozen": all(not p.requires_grad for p in model.parameters()),
        "trace_source": cfg.trace_source,
        "injection_point": wrapper.injector.injection_point,
        "tap_layers": wrapper.tap_layer_ids,
        "inject_layers": wrapper.inject_layer_ids,
        "pass": len(changed) == 0 and all(n.startswith("steering.") for n, _ in trainable),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
