
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TraceFrame:
    step_index: int
    hidden: Optional[torch.Tensor] = None
    kv: Optional[torch.Tensor] = None
    logits_summary: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    layer_id: Optional[int] = None


class TraceBank:
    def __init__(self, max_steps="all", detach=True):
        self.max_steps = max_steps
        self.detach = detach
        self.frames: list[TraceFrame] = []

    def reset(self):
        self.frames.clear()

    def append(self, frame: TraceFrame):
        if self.detach:
            frame = TraceFrame(
                step_index=frame.step_index,
                hidden=None if frame.hidden is None else frame.hidden.detach(),
                kv=None if frame.kv is None else frame.kv.detach(),
                logits_summary=None if frame.logits_summary is None else frame.logits_summary.detach(),
                entropy=None if frame.entropy is None else frame.entropy.detach(),
                layer_id=frame.layer_id,
            )
        self.frames.append(frame)
        if self.max_steps != "all":
            m = int(self.max_steps)
            if len(self.frames) > m:
                self.frames = self.frames[-m:]

    def get_before_step(self, k=None):
        return self.frames

    def as_tensor(self):
        vals = [f.hidden for f in self.frames if f.hidden is not None]
        if not vals:
            return None
        vals = [v for v in vals if v.dim() == 3]
        if not vals:
            return None
        # [B, T_trace, bd, hidden_dim]
        return torch.stack(vals, dim=1)

    def __len__(self):
        return len(self.frames)
