"""Chunked VAE decode parity vs upstream PT (REAL weights, CPU).

Requires the downloaded checkpoint; skipped if weights are absent.
Locks: frame count 1+(T-1)*4 and numerical parity of every frame, including
the first-chunk 'Rep' boundary that mlx-video's whole-sequence decode gets
wrong (4*T frames, divergent head).
"""
from pathlib import Path

import numpy as np
import pytest
import torch

import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_video.models.wan_2.vae import WanVAE as MxVAE

from scail2_mlx.utils.vae_stream import decode_chunked

from ._helpers import assert_parity

REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_PTH = REPO_ROOT / "weights" / "SCAIL-2" / "Wan2.1_VAE.pth"
VAE_MLX = REPO_ROOT / "weights" / "mlx" / "vae.safetensors"

pytestmark = pytest.mark.skipif(
    not (VAE_PTH.exists() and VAE_MLX.exists()), reason="VAE weights not downloaded"
)


@pytest.fixture(scope="module")
def vaes(pt_scail2):  # pt_scail2 ensures wanref package + stubs exist
    from tests.parity.conftest import _load_upstream

    vae_ref = _load_upstream("vae")
    pt_vae = vae_ref.WanVAE(vae_pth=str(VAE_PTH), device="cpu", dtype=torch.float32)

    mx_vae = MxVAE(z_dim=16, encoder=True)
    mx_vae.update(tree_unflatten(list(mx.load(str(VAE_MLX)).items())))
    mx.eval(mx_vae.parameters())
    return pt_vae, mx_vae


@pytest.mark.parametrize("t_lat", [1, 5])
def test_decode_chunked_parity(vaes, t_lat):
    pt_vae, mx_vae = vaes
    z = np.random.default_rng(3).standard_normal((16, t_lat, 8, 8)).astype("float32") * 0.8

    with torch.no_grad():
        pt_out = pt_vae.decode([torch.from_numpy(z)])[0]

    mx_out = decode_chunked(mx_vae, mx.array(z)[None])[0]

    assert pt_out.shape[1] == 1 + (t_lat - 1) * 4
    assert_parity(pt_out, mx_out, 5e-4, name=f"decode_chunked T={t_lat}")


def test_encode_parity(vaes):
    pt_vae, mx_vae = vaes
    x = np.random.default_rng(4).standard_normal((3, 13, 32, 32)).astype("float32") * 0.3

    with torch.no_grad():
        pt_lat = pt_vae.encode([torch.from_numpy(x)])[0]
    mx_lat = mx_vae.encode(mx.array(x)[None])[0]
    assert_parity(pt_lat, mx_lat, 5e-4, name="encode")
