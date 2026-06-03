"""Dual-stream mask unit test (verifies N2N/N2C/C2C + auto-truncation). No GPU needed.

    python tests/test_mask.py
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fast_dvlm.mask import stream_meta, asymmetric_allowed


def build(labels, vision, bd):
    meta = stream_meta(labels, vision, bd)
    turn = meta["turn_idx"]; tp = meta["text_positions"]; Lt = meta["L_text"]; total = meta["total"]
    qi = torch.arange(total).view(total, 1).expand(total, total)
    ki = torch.arange(total).view(1, total).expand(total, total)
    allowed = asymmetric_allowed(qi, ki, turn[tp], turn, Lt)
    return meta, allowed


def test_dual_stream():
    # [V V V | P P P | R R R R R]  (vision3, prompt3, response5), block size 2
    L = 11
    labels = torch.full((L,), -100, dtype=torch.int64)
    labels[6:11] = torch.tensor([101, 102, 103, 104, 105])
    vision = torch.zeros(L, dtype=torch.bool); vision[0:3] = True
    meta, a = build(labels, vision, 2)
    assert meta["L_text"] == 8 and meta["L_clean"] == 11 and meta["total"] == 19
    assert meta["n_blocks"] == 3, meta["n_blocks"]              # ceil(5/2)=3 -> auto-truncation
    Lt = meta["L_text"]
    clean = a[Lt:, Lt:]
    assert torch.equal(clean, torch.tril(torch.ones_like(clean))), "C2C must be plain causal"
    # clean never attends to the noisy half
    assert not a[Lt:, :Lt].any(), "clean must not attend noisy"
    print("test_dual_stream: PASS  (L_text=8 L_clean=11 total=19 n_blocks=3; C2C causal; clean!->noisy)")


if __name__ == "__main__":
    test_dual_stream()
