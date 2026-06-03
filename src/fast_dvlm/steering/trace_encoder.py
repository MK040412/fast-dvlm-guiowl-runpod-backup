
from __future__ import annotations
import torch
from torch import nn


class TraceEncoder(nn.Module):
    def __init__(self, hidden_dim: int, steer_dim: int, bd: int, max_steps: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steer_dim = steer_dim
        self.bd = bd
        self.trace_proj = nn.Linear(hidden_dim, steer_dim)
        self.empty_trace = nn.Parameter(torch.zeros(1, 1, steer_dim))
        self.pos_embed = nn.Embedding(bd, steer_dim)
        self.step_embed = nn.Embedding(max_steps, steer_dim)
        self.norm = nn.LayerNorm(steer_dim)

    def forward(self, trace_tensor, *, batch_size: int, bd: int, step_index: int = 0, device=None, dtype=None):
        device = device or self.empty_trace.device
        dtype = dtype or self.empty_trace.dtype
        pos = torch.arange(bd, device=device).clamp(max=self.bd - 1)
        pos_e = self.pos_embed(pos).to(dtype).unsqueeze(0)
        step_id = torch.tensor([min(int(step_index), self.step_embed.num_embeddings - 1)], device=device)
        step_e = self.step_embed(step_id).to(dtype).view(1, 1, -1)
        if trace_tensor is None:
            cur = self.empty_trace.to(device=device, dtype=dtype).expand(batch_size, bd, -1)
            cur = cur + pos_e + step_e
            return self.norm(cur), None
        trace_tensor = trace_tensor.to(device=device, dtype=dtype)
        enc = self.trace_proj(trace_tensor)  # [B, T, bd, D]
        cur = enc.mean(dim=1) + pos_e + step_e
        return self.norm(cur), enc
