
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class SteeringConfig:
    bd: int = 32
    tap_layers: str = "last4"
    inject_layers: str = "last8"
    trace_source_priority: list[str] = field(default_factory=lambda: ["kv", "hidden", "logits_entropy_topk"])
    allow_hidden_fallback: bool = True
    fail_if_no_trace: bool = False
    trace_source: str = "hidden_fallback"
    injection_point: str = "mlp_output"
    action_token_slice: str = "last_bd"

    steer_dim: int = 512
    mla_dim: int = 256
    num_heads: int = 8
    ssd_layers: int = 3
    mla_layers: int = 1
    n_steer_blocks: int = 2

    gate_type: str = "gaussian"
    gate_center: float = 0.35
    gate_width: float = 0.15
    learn_gate: bool = False
    zero_init_residual: bool = True

    lambda_action: float = 0.1
    lambda_coord: float = 0.05
    lambda_res: float = 1e-4
    struct_weight: float = 1.5
    coord_token_weight: float = 2.0

    trace_detach: bool = True
    max_trace_steps: str = "all"
    train_steps_per_block: int = 2
    mode_selection_window_center: float = 0.35
    mode_selection_window_width: float = 0.15

    hidden_dim: int | None = None
    vocab_size: int | None = None
    base_model: str | None = None

    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def load_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
