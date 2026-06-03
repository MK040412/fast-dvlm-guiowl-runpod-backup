
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class SSDLinearBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.trace_proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.raw_lambda = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, trace_seq=None):
        h = self.norm(x)
        if trace_seq is not None:
            # Cheap SSD-style denoise-time recurrence over trace frames.
            state = torch.zeros_like(h)
            lam = torch.tanh(self.raw_lambda) * 0.5 + 0.5
            for t in range(trace_seq.shape[1]):
                state = lam * state + self.trace_proj(trace_seq[:, t])
            h = h + state / max(int(trace_seq.shape[1]), 1)
        return x + self.ff(h)


class MLAAnchorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mla_dim: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.kv_down = nn.Linear(dim, mla_dim)
        self.kv_up = nn.Linear(mla_dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out = nn.Linear(dim, dim)
        self.learned_anchor = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, trace_seq=None):
        b = x.shape[0]
        anchors = [self.learned_anchor.expand(b, -1, -1)]
        if trace_seq is not None:
            anchors.append(trace_seq.mean(dim=(1, 2), keepdim=False).unsqueeze(1))
        kv = torch.cat(anchors, dim=1)
        kv = self.kv_up(self.kv_down(kv))
        y, _ = self.attn(self.norm_q(x), kv, kv, need_weights=False)
        return x + self.out(y)


class KimiTraceSteering(nn.Module):
    def __init__(self, dim: int, num_heads: int, mla_dim: int, ssd_layers: int = 3, mla_layers: int = 1, n_blocks: int = 2):
        super().__init__()
        blocks = []
        for _ in range(n_blocks):
            for _ in range(ssd_layers):
                blocks.append(SSDLinearBlock(dim))
            for _ in range(mla_layers):
                blocks.append(MLAAnchorBlock(dim, num_heads, mla_dim))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, trace_seq=None):
        for block in self.blocks:
            x = block(x, trace_seq)
        return self.norm(x)
