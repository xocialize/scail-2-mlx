"""Batched CFG equivalence: one B=2 forward [cond, uncond] must equal the two
sequential B=1 forwards. Pure-MLX self-consistency at the tiny config (CPU)."""
import numpy as np
import torch

import mlx.core as mx

from scail2_mlx.modules.model_scail2 import SCAIL2Model
from scail2_mlx.utils.weights import load_state_dict_np

from ._helpers import assert_parity, make_seeded_input
from .test_model_forward_parity import TINY, T_, H_, W_, _make_inputs


def test_batched_cfg_equals_sequential(pt_scail2):
    torch.manual_seed(0)
    pt_model = pt_scail2.SCAIL2Model(**TINY).eval().float()
    state = {k: v.detach().numpy() for k, v in pt_model.state_dict().items()}
    rng = np.random.default_rng(99)
    for key, arr in state.items():
        if np.abs(arr).sum() == 0 or arr.std() == 0:
            state[key] = (0.02 * rng.standard_normal(arr.shape)).astype(arr.dtype)

    model = SCAIL2Model(**TINY)
    load_state_dict_np(model, state, strict=True)

    inp = _make_inputs(False)
    ctx_cond = mx.array(inp["context"][0])
    ctx_null = mx.array(make_seeded_input((5, TINY["text_dim"]), seed=21))
    clip_fea = mx.array(inp["clip_fea"])

    common = dict(
        pose_latents=[mx.array(inp["pose_latents"][0])],
        driving_masks=[mx.array(inp["driving_masks"][0])],
        ref_latents=[mx.array(inp["ref_latents"][0])],
        ref_masks=[mx.array(inp["ref_masks"][0])],
        seq_len=int(1e10),
        replace_flag=False,
        history_mask=None,
    )
    x = mx.array(inp["x"][0])

    cond = model(
        x=[x], t=mx.array([500.0]), context=[ctx_cond], clip_fea=clip_fea, **common
    )[0]
    uncond = model(
        x=[x], t=mx.array([500.0]), context=[ctx_null], clip_fea=clip_fea, **common
    )[0]

    batched_common = {
        k: (v * 2 if isinstance(v, list) else v) for k, v in common.items()
    }
    preds = model(
        x=[x, x],
        t=mx.array([500.0, 500.0]),
        context=[ctx_cond, ctx_null],
        clip_fea=mx.concatenate([clip_fea, clip_fea], axis=0),
        **batched_common,
    )

    assert_parity(np.array(cond), preds[0], 1e-5, name="batched cond")
    assert_parity(np.array(uncond), preds[1], 1e-5, name="batched uncond")
