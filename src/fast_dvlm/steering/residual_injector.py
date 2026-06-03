
from __future__ import annotations
import re
from dataclasses import dataclass
import torch
from torch import nn


_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")
_BAD = ("visual", "vision", "vit", "merger")
_MLP_NAMES = {"mlp", "ffn", "feed_forward", "feedforward"}


def _get_submodule(root: nn.Module, name: str):
    mod = root
    for part in name.split('.'):
        mod = getattr(mod, part)
    return mod


def _set_submodule(root: nn.Module, name: str, value: nn.Module):
    parts = name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _layer_id(name: str):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def find_mlp_modules(model: nn.Module):
    found = {}
    for name, module in model.named_modules():
        low = name.lower()
        if any(b in low for b in _BAD):
            continue
        lid = _layer_id(name)
        if lid is None:
            continue
        last = name.rsplit('.', 1)[-1].lower()
        if last in _MLP_NAMES:
            found[lid] = (name, module)
    return dict(sorted(found.items()))


def select_layer_ids(all_ids, spec: str):
    ids = list(sorted(all_ids))
    if not ids:
        return []
    spec = str(spec)
    if spec.startswith("last"):
        n = int(spec[4:])
        return ids[-n:]
    if spec == "all":
        return ids
    return [int(x) for x in spec.split(',') if x.strip()]


class ResidualAddWrapper(nn.Module):
    def __init__(self, module: nn.Module, layer_id: int, owner: "ResidualInjector"):
        super().__init__()
        self.module = module
        self.layer_id = layer_id
        self.owner = owner

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        res = self.owner.residuals.get(self.layer_id)
        if res is None:
            return out
        if isinstance(out, tuple):
            main = out[0]
            rest = out[1:]
        else:
            main = out
            rest = None
        if not torch.is_tensor(main) or main.dim() != 3:
            return out
        bd = res.shape[1]
        res = res.to(device=main.device, dtype=main.dtype)
        if main.shape[0] == res.shape[0]:
            if main.shape[1] < bd or main.shape[2] != res.shape[2]:
                raise RuntimeError(f"bad residual shape layer={self.layer_id} out={tuple(main.shape)} res={tuple(res.shape)}")
            y = main.clone()
            y[:, -bd:, :] = y[:, -bd:, :] + res
        elif main.shape[1] == res.shape[0]:
            if main.shape[0] < bd or main.shape[2] != res.shape[2]:
                raise RuntimeError(f"bad residual shape layer={self.layer_id} out={tuple(main.shape)} res={tuple(res.shape)}")
            y = main.clone()
            y[-bd:, :, :] = y[-bd:, :, :] + res.transpose(0, 1)
        else:
            raise RuntimeError(f"cannot align residual layer={self.layer_id} out={tuple(main.shape)} res={tuple(res.shape)}")
        if rest is not None:
            return (y,) + rest
        return y


@dataclass
class InjectMap:
    available_layers: list[int]
    inject_layer_ids: list[int]
    tap_layer_ids: list[int]
    injection_point: str
    module_names: dict


class ResidualInjector:
    def __init__(self, model: nn.Module, inject_layer_ids):
        self.model = model
        self.inject_layer_ids = list(inject_layer_ids)
        self.residuals: dict[int, torch.Tensor] = {}
        self.mlp_modules = find_mlp_modules(model)
        self.wrapped = {}
        self.injection_point = "mlp_output"

    def register(self):
        for lid in self.inject_layer_ids:
            if lid not in self.mlp_modules:
                continue
            name, module = self.mlp_modules[lid]
            if isinstance(module, ResidualAddWrapper):
                self.wrapped[lid] = (name, module)
                continue
            wrapper = ResidualAddWrapper(module, lid, self)
            _set_submodule(self.model, name, wrapper)
            self.wrapped[lid] = (name, wrapper)
        if len(self.wrapped) != len(self.inject_layer_ids):
            self.injection_point = "mlp_output_partial"
        return self

    def set_residuals(self, residuals: dict[int, torch.Tensor]):
        self.residuals = residuals

    def clear_residuals(self):
        self.residuals = {}

    def module_names(self):
        return {str(k): v[0] for k, v in self.wrapped.items()}
