"""SCAIL 3-segment RoPE parity: PT (complex float64) vs MLX (cos/sin float32).

Covers both shift regimes:
  - animation: video/pose T-shift 1, ref at T 0
  - replace:   ref H-shift 120, pose W-shift 120
and the pose branch's avg-pooled (k=2, s=2) complex frequencies.
"""
import pytest
import torch

import mlx.core as mx

from scail2_mlx.modules import model_scail2 as mx_m

from ._helpers import assert_parity, make_seeded_input

B, N, D = 2, 4, 32  # heads=4, head_dim=32 -> c=16, split [6, 5, 5]
F_, H_, W_ = 4, 8, 8


def _pt_freqs(pt_scail2, d=D):
    return torch.cat(
        [
            pt_scail2.rope_params(8192, d - 4 * (d // 6)),
            pt_scail2.rope_params(8192, 2 * (d // 6)),
            pt_scail2.rope_params(8192, 2 * (d // 6)),
        ],
        dim=1,
    )


def _mx_freqs(d=D):
    f_t = mx_m.rope_params(8192, d - 4 * (d // 6))
    f_h = mx_m.rope_params(8192, 2 * (d // 6))
    f_w = mx_m.rope_params(8192, 2 * (d // 6))
    return (
        mx.concatenate([f_t[0], f_h[0], f_w[0]], axis=1),
        mx.concatenate([f_t[1], f_h[1], f_w[1]], axis=1),
    )


def _shift_kwargs(replace_flag: bool):
    return dict(
        rope_T=F_,
        rope_H=H_,
        rope_W=W_,
        rope_T_shift={
            "ref": 0,
            "pose": 0 if replace_flag else 1,
            "video": 0 if replace_flag else 1,
        },
        rope_H_shift={
            "ref": 120 if replace_flag else 0,
            "pose": 0,
            "video": 0,
        },
        rope_W_shift={"ref": 0, "pose": 120, "video": 0},
    )


def _run_segment(pt_scail2, fn_name: str, seq_len: int, replace_flag: bool, seed: int):
    x_np = make_seeded_input((B, seq_len, N, D), seed=seed)
    kw = _shift_kwargs(replace_flag)

    pt_out = getattr(pt_scail2, fn_name)(
        torch.from_numpy(x_np), _pt_freqs(pt_scail2), **kw
    )
    mx_out = getattr(mx_m, fn_name)(mx.array(x_np), freqs=_mx_freqs(), **kw)
    assert_parity(pt_out, mx_out, 1e-4, name=f"{fn_name} replace={replace_flag}")


@pytest.mark.parametrize("replace_flag", [False, True])
@pytest.mark.parametrize(
    "fn_name,seq_len",
    [
        ("rope_apply_ref", 1 * H_ * W_),
        ("rope_apply_video", F_ * H_ * W_),
        ("rope_apply_pose", F_ * (H_ // 2) * (W_ // 2)),
    ],
)
def test_rope_segment(pt_scail2, fn_name, seq_len, replace_flag):
    _run_segment(pt_scail2, fn_name, seq_len, replace_flag, seed=hash(fn_name) % 1000)


@pytest.mark.parametrize("replace_flag", [False, True])
def test_rope_apply_scail_full(pt_scail2, replace_flag):
    ref_len = H_ * W_
    vid_len = F_ * H_ * W_
    pose_len = F_ * (H_ // 2) * (W_ // 2)
    x_np = make_seeded_input((B, ref_len + vid_len + pose_len, N, D), seed=7)

    kw = _shift_kwargs(replace_flag)
    kw.update(ref_length=ref_len, seq_length=vid_len, pose_length=pose_len)

    pt_out = pt_scail2.rope_apply_scail(
        torch.from_numpy(x_np), freqs=_pt_freqs(pt_scail2), **kw
    )
    mx_out = mx_m.rope_apply_scail(mx.array(x_np), freqs=_mx_freqs(), **kw)
    assert_parity(pt_out, mx_out, 1e-4, name=f"rope_apply_scail replace={replace_flag}")
