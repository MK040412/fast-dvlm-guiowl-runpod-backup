"""Fast-dVLM: convert GUI-Owl-1.5-2B (Qwen3-VL) into a block-diffusion VLM.

Faithful port of NVlabs/Fast-dLLM `fast_dvlm` (arXiv 2604.06832) to the Qwen3-VL
architecture, with single-device throughput optimisations (frozen+cached vision,
FlexAttention block-sparse mask, grad-ckpt off, torch.compile).
"""
from .mask import (compute_response_block_idx, asymmetric_allowed, stream_meta,
                   build_dense_mask, build_block_mask)
from .forward import dual_stream_loss
from .decode import ar_generate, dvlm_generate

__all__ = ["compute_response_block_idx", "asymmetric_allowed", "stream_meta",
           "build_dense_mask", "build_block_mask", "dual_stream_loss",
           "ar_generate", "dvlm_generate"]
__version__ = "0.1.0"
