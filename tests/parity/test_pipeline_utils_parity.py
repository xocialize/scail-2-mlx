"""Pipeline helper parity: _half_bilinear == torch bilinear 0.5x downscale."""
import pytest
import torch
import torch.nn.functional as F

import mlx.core as mx

from scail2_mlx.scail import _half_bilinear

from ._helpers import assert_parity, make_seeded_input


@pytest.mark.parametrize("shape", [(5, 3, 64, 96), (3, 17, 128, 128)])
def test_half_bilinear_parity(shape):
    x_np = make_seeded_input(shape, seed=13)
    pt_out = F.interpolate(
        torch.from_numpy(x_np), scale_factor=0.5, mode="bilinear", align_corners=False
    )
    mx_out = _half_bilinear(mx.array(x_np))
    assert_parity(pt_out, mx_out, 1e-6, name=f"half_bilinear {shape}")
