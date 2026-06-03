
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from torch import nn

from .config import SteeringConfig
from .trace_bank import TraceBank, TraceFrame
from .trace_encoder import TraceEncoder
from .kimi_trace_steering import KimiTraceSteering
from .residual_injector import ResidualInjector, find_mlp_modules, select_layer_ids


def _text_config(model):
    cfg = getattr(model.config, "text_config", model.config)
    return cfg


def infer_hidden_dim(model):
    cfg = _text_config(model)
    for key in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    emb = model.get_input_embeddings().weight
    return int(emb.shape[1])


def infer_vocab_size(model):
    cfg = _text_config(model)
    if hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    return int(model.get_input_embeddings().weight.shape[0])


class SteeringWrapper(nn.Module):
    def __init__(self, base_model, config: SteeringConfig):
        super().__init__()
        object.__setattr__(self, "base_model", base_model)
        self.config = config
        self.hidden_dim = config.hidden_dim or infer_hidden_dim(base_model)
        self.vocab_size = config.vocab_size or infer_vocab_size(base_model)
        self.config.hidden_dim = self.hidden_dim
        self.config.vocab_size = self.vocab_size

        mlps = find_mlp_modules(base_model)
        all_ids = list(mlps.keys())
        self.tap_layer_ids = select_layer_ids(all_ids, config.tap_layers)
        self.inject_layer_ids = select_layer_ids(all_ids, config.inject_layers)
        self.trace_bank = TraceBank(max_steps=config.max_trace_steps, detach=config.trace_detach)

        self.trace_encoder = TraceEncoder(self.hidden_dim, config.steer_dim, config.bd)
        self.steering = KimiTraceSteering(
            config.steer_dim, config.num_heads, config.mla_dim,
            ssd_layers=config.ssd_layers, mla_layers=config.mla_layers, n_blocks=config.n_steer_blocks,
        )
        self.residual_heads = nn.ModuleDict({str(lid): nn.Linear(config.steer_dim, self.hidden_dim) for lid in self.inject_layer_ids})
        if config.zero_init_residual:
            for head in self.residual_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        object.__setattr__(self, "injector", ResidualInjector(base_model, self.inject_layer_ids).register())
        self.capture_handles = []
        self.capture_active = False
        self.last_residual_norm_mean = 0.0
        self.last_residual_norm_max = 0.0
        self.last_gate = 0.0
        self.last_res_loss = None

    def freeze_base(self):
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _tensor_from_output(self, output):
        if isinstance(output, tuple):
            output = output[0]
        return output if torch.is_tensor(output) else None

    def begin_trace_capture(self):
        self.trace_bank.reset()
        self.capture_active = True
        mlps = self.injector.mlp_modules
        step_base = 0
        for lid in self.tap_layer_ids:
            if lid not in mlps:
                continue
            module = mlps[lid][1]
            def make_hook(layer_id):
                def hook(mod, inp, out):
                    if not self.capture_active:
                        return
                    t = self._tensor_from_output(out)
                    if t is None or t.dim() != 3:
                        return
                    bd = self.config.bd
                    if t.shape[1] >= bd:
                        h = t[:, -bd:, :]
                    elif t.shape[0] >= bd:
                        h = t[-bd:, :, :].transpose(0, 1)
                    else:
                        return
                    self.trace_bank.append(TraceFrame(step_index=len(self.trace_bank.frames), hidden=h.detach(), layer_id=layer_id))
                return hook
            self.capture_handles.append(module.register_forward_hook(make_hook(lid)))
            step_base += 1

    def end_trace_capture(self):
        self.capture_active = False
        for h in self.capture_handles:
            h.remove()
        self.capture_handles = []
        if len(self.trace_bank) == 0 and self.config.fail_if_no_trace:
            raise RuntimeError("no trace frames captured")
        return self.trace_bank

    def gate_value(self, progress=None):
        progress = self.config.mode_selection_window_center if progress is None else float(progress)
        if self.config.gate_type == "gaussian":
            width = max(float(self.config.gate_width), 1e-6)
            return float(math.exp(-0.5 * ((progress - self.config.gate_center) / width) ** 2))
        return 1.0

    def prepare_residuals(self, *, batch_size=None, device=None, dtype=None, progress=None):
        trace = self.trace_bank.as_tensor()
        if trace is not None:
            batch_size = int(trace.shape[0])
            device = device or trace.device
            dtype = dtype or trace.dtype
        if batch_size is None:
            batch_size = 2
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        x, trace_seq = self.trace_encoder(trace, batch_size=batch_size, bd=self.config.bd, step_index=0, device=device, dtype=dtype)
        z = self.steering(x, trace_seq)
        gate = self.gate_value(progress)
        residuals = {}
        norms = []
        res_loss = None
        for lid, head in self.residual_heads.items():
            r = head(z) * gate
            residuals[int(lid)] = r
            norms.append(r.float().norm(dim=-1).mean())
            term = r.float().pow(2).mean()
            res_loss = term if res_loss is None else res_loss + term
        self.injector.set_residuals(residuals)
        if norms:
            ns = torch.stack(norms)
            self.last_residual_norm_mean = float(ns.mean().detach().cpu())
            self.last_residual_norm_max = float(ns.max().detach().cpu())
        else:
            self.last_residual_norm_mean = 0.0
            self.last_residual_norm_max = 0.0
        self.last_gate = gate
        self.last_res_loss = res_loss if res_loss is not None else torch.tensor(0.0, device=device)
        return residuals

    def clear_residuals(self):
        self.injector.clear_residuals()
        self.last_res_loss = None

    def adapter_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def save_adapter(self, out_dir, training_args=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
            save_file(self.adapter_state_dict(), str(out / "steering_model.safetensors"))
            weight_file = "steering_model.safetensors"
        except Exception:
            torch.save(self.adapter_state_dict(), out / "pytorch_model.bin")
            weight_file = "pytorch_model.bin"
        self.config.injection_point = self.injector.injection_point
        self.config.save_json(out / "steering_config.json")
        inject_map = {
            "tap_layer_ids": self.tap_layer_ids,
            "inject_layer_ids": self.inject_layer_ids,
            "injection_point": self.injector.injection_point,
            "module_names": self.injector.module_names(),
            "weight_file": weight_file,
        }
        (out / "inject_map.json").write_text(json.dumps(inject_map, indent=2, sort_keys=True))
        if training_args is not None:
            (out / "training_args.json").write_text(json.dumps(training_args, indent=2, sort_keys=True, default=str))
        (out / "README.md").write_text(
            "# bd32 Fast-dVLM Denoise-Trace Steering Adapter\n\n"
            f"This is a steering-only adapter for `{self.config.base_model}`.\n\n"
            "The GUI-Owl/Fast-dVLM base model must be loaded separately and kept frozen. "
            "This directory intentionally does not duplicate base checkpoint weights.\n"
        )
