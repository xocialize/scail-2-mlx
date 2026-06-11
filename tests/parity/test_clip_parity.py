"""CLIP visual tower parity vs upstream (CPU oracle, fp32).

Covers the ViT forward (both full and use_31_block), the torch-kernel bicubic
resize matrices, and the CLIPModel.visual() end-to-end path including
[-1,1] -> CLIP-normalized preprocessing.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mlx.core as mx

from scail2_mlx.modules import clip as mx_clip
from scail2_mlx.utils.weights import load_state_dict_np

from ._helpers import assert_parity, make_seeded_input

TINY_VIT = dict(
    image_size=28,
    patch_size=14,
    dim=64,
    mlp_ratio=4,
    out_dim=32,
    num_heads=4,
    num_layers=3,
    pool_type="token",
    pre_norm=True,
    post_norm=False,
    activation="gelu",
)

_PT_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_PT_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


@pytest.mark.parametrize("use_31_block", [True, False])
def test_vit_forward_parity(pt_clip, use_31_block):
    torch.manual_seed(0)
    pt_model = pt_clip.VisionTransformer(**TINY_VIT).eval().float()
    state = {k: v.detach().numpy() for k, v in pt_model.state_dict().items()}

    mx_model = mx_clip.VisionTransformer(**TINY_VIT)
    load_state_dict_np(mx_model, state, strict=True)

    x_np = make_seeded_input((2, 3, 28, 28), seed=3)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(x_np), use_31_block=use_31_block)
    mx_out = mx_model(mx.array(x_np), use_31_block=use_31_block)
    assert_parity(pt_out, mx_out, 1e-4, name=f"ViT use_31_block={use_31_block}")


@pytest.mark.parametrize("hw", [(123, 77), (224, 224), (640, 360)])
def test_bicubic_resize_parity(hw):
    h, w = hw
    x_np = make_seeded_input((2, 3, h, w), seed=5)
    pt_out = F.interpolate(
        torch.from_numpy(x_np), size=(224, 224), mode="bicubic", align_corners=False
    )

    mh = mx_clip.bicubic_resize_matrix(h, 224)
    mw = mx_clip.bicubic_resize_matrix(w, 224)
    mx_out = mx.matmul(mx.matmul(mh, mx.array(x_np)), mw.T)
    # 5e-4: identical taps/weights (verified via one-hot probes); residual is
    # fp32 rounding-order skew between torch's two separable passes and the
    # fp64-derived matrix composition, growing with input size
    assert_parity(pt_out, mx_out, 5e-4, name=f"bicubic {hw}->224")


def test_clip_visual_e2e_tiny(pt_clip):
    torch.manual_seed(1)
    pt_model = pt_clip.VisionTransformer(**TINY_VIT).eval().float()
    state = {k: v.detach().numpy() for k, v in pt_model.state_dict().items()}

    mx_model = mx_clip.VisionTransformer(**TINY_VIT)
    load_state_dict_np(mx_model, state, strict=True)

    # [3, 1, H, W] in [-1, 1], like the pipeline's img[:, None, :, :]
    img_np = np.tanh(make_seeded_input((3, 1, 90, 70), seed=6))

    # PT side: replicate upstream CLIPModel.visual() preprocessing exactly
    with torch.no_grad():
        videos = torch.cat(
            [
                F.interpolate(
                    torch.from_numpy(img_np).transpose(0, 1),
                    size=(28, 28),
                    mode="bicubic",
                    align_corners=False,
                )
            ]
        )
        videos = videos.mul(0.5).add(0.5)
        videos = (videos - _PT_MEAN) / _PT_STD
        pt_out = pt_model(videos, use_31_block=True)

    # MLX side: production wrapper (image_size derived from the model)
    wrapper = mx_clip.CLIPModel.__new__(mx_clip.CLIPModel)
    wrapper.dtype = mx.float32
    wrapper.model = type("M", (), {"visual": mx_model, "image_size": 28})()
    wrapper.size = 28
    wrapper._resize_cache = {}
    wrapper._mean = mx.array(mx_clip._CLIP_MEAN).reshape(1, 3, 1, 1)
    wrapper._std = mx.array(mx_clip._CLIP_STD).reshape(1, 3, 1, 1)
    mx_out = wrapper.visual([mx.array(img_np)])

    assert_parity(pt_out, mx_out, 1e-4, name="CLIPModel.visual e2e")
