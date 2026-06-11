"""extract_and_compress_mask_to_latent parity vs upstream (CPU oracle).

Upstream scail_utils.py imports decord/torchvision at module level; both are
stubbed since only the pure-tensor function is exercised.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mlx.core as mx

from scail2_mlx.utils.scail_utils import extract_and_compress_mask_to_latent

from ._helpers import assert_parity

UPSTREAM = (
    Path(__file__).resolve().parents[2] / "refs" / "SCAIL-2" / "wan" / "utils" / "scail_utils.py"
)


@pytest.fixture(scope="session")
def pt_scail_utils():
    for name in ("decord", "torchvision", "torchvision.transforms",
                 "torchvision.transforms.functional"):
        mod = types.ModuleType(name)
        sys.modules.setdefault(name, mod)
    sys.modules["decord"].VideoReader = object
    sys.modules["decord"].bridge = types.SimpleNamespace(set_bridge=lambda *_: None)
    tv_t = sys.modules["torchvision.transforms"]
    tv_t.InterpolationMode = types.SimpleNamespace(BICUBIC="bicubic")
    tv_t.Compose = object
    tv_t.ToTensor = object
    tv_t.functional = sys.modules["torchvision.transforms.functional"]
    sys.modules["torchvision.transforms.functional"].center_crop = lambda *a, **k: None
    sys.modules["torchvision.transforms.functional"].resize = lambda *a, **k: None
    sys.modules["torchvision"].transforms = tv_t

    spec = importlib.util.spec_from_file_location("scail_utils_ref", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("shape", [(3, 9, 64, 64), (3, 17, 32, 96)])
def test_mask_compress_parity(pt_scail_utils, shape):
    rng = np.random.default_rng(11)
    # mimic RGB mask pixels: mostly saturated channel values around the
    # on-threshold so both sides of the > comparison are exercised
    raw = rng.choice([0, 200, 230, 255], size=shape).astype(np.float32)
    mask = (raw - 127.5) / 127.5

    pt_out = pt_scail_utils.extract_and_compress_mask_to_latent(
        torch.from_numpy(mask), additional_spatial_downsample=1
    )
    mx_out = extract_and_compress_mask_to_latent(
        mx.array(mask), additional_spatial_downsample=1
    )
    assert_parity(pt_out, mx_out, 1e-6, name=f"mask_compress {shape}")
