
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from fast_dvlm.steering.residual_injector import find_mlp_modules, select_layer_ids
    from fast_dvlm.steering.wrapper import infer_hidden_dim, infer_vocab_size
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    mlps = find_mlp_modules(model)
    candidate_attn = []
    candidate_kv = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(x in low for x in ["visual", "vision"]):
            continue
        if "attn" in low or "attention" in low:
            candidate_attn.append(name)
        if low.endswith("k_proj") or low.endswith("v_proj") or "qkv" in low:
            candidate_kv.append(name)
    try:
        tok = AutoProcessor.from_pretrained(args.model).tokenizer
        special = getattr(tok, "special_tokens_map", {})
    except Exception:
        special = {}
    report = {
        "model": args.model,
        "hidden_dim": infer_hidden_dim(model),
        "vocab_size": infer_vocab_size(model),
        "num_mlp_layers": len(mlps),
        "available_layers": list(mlps.keys()),
        "tap_last4": select_layer_ids(mlps.keys(), "last4"),
        "inject_last8": select_layer_ids(mlps.keys(), "last8"),
        "candidate_mlp_modules": {str(k): v[0] for k, v in mlps.items()},
        "candidate_attention_modules_head": candidate_attn[:80],
        "candidate_kv_modules_head": candidate_kv[:80],
        "supports_output_hidden_states": hasattr(model.config, "output_hidden_states"),
        "tokenizer_special_tokens": special,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

if __name__ == "__main__":
    main()
