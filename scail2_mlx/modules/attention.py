# MLX stand-in for refs/SCAIL-2/wan/modules/attention.py `flash_attention`.
# Same call signature where exercised by model_scail2.py. SCAIL-2 always runs
# global attention with uniform sequence lengths, so k_lens / window_size are
# accepted but must be trivial.
import math

import mlx.core as mx

__all__ = ["flash_attention"]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    window_size=(-1, -1),
):
    """
    q: [B, Lq, Nq, C1].  k: [B, Lk, Nk, C1].  v: [B, Lk, Nk, C2].
    Returns [B, Lq, Nq, C2], matching the upstream wrapper's output layout.
    """
    assert tuple(window_size) == (-1, -1), "SCAIL-2 uses global attention only"

    scale = 1.0 / math.sqrt(q.shape[-1])
    out = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        scale=scale,
    )
    return out.transpose(0, 2, 1, 3)
