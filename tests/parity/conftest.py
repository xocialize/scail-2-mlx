"""Load upstream PT reference modules (refs/SCAIL-2) importable on CPU.

Upstream `wan.modules.attention.flash_attention` asserts CUDA, and the `wan`
package __init__ drags in decord/distributed; clip.py additionally imports
torchvision and tokenizers. Modules are loaded individually under a synthetic
`wanref` package with CPU/no-op stand-ins pre-injected.
"""
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

import mlx.core as mx

# Parity oracles compare against PT on CPU. MLX's Metal fp32 GEMM on this
# machine (mlx 0.31.2) runs at ~1e-3 relative precision (TF32-class fast
# path), which swamps 1e-4 thresholds; the CPU stream uses Accelerate and is
# exact. Production inference runs bf16 on GPU where this is immaterial.
mx.set_default_device(mx.cpu)

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_MODULES = REPO_ROOT / "refs" / "SCAIL-2" / "wan" / "modules"


def _cpu_flash_attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None,
    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
    dtype=None, version=None,
):
    import torch.nn.functional as F

    assert tuple(window_size) == (-1, -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=softmax_scale, is_causal=causal,
    )
    return out.transpose(1, 2)


def _ensure_wanref():
    if "wanref" in sys.modules:
        return
    pkg = types.ModuleType("wanref")
    pkg.__path__ = []
    modules_pkg = types.ModuleType("wanref.modules")
    modules_pkg.__path__ = []
    attn = types.ModuleType("wanref.modules.attention")
    attn.flash_attention = _cpu_flash_attention
    tok = types.ModuleType("wanref.modules.tokenizers")
    tok.HuggingfaceTokenizer = object
    sys.modules["wanref"] = pkg
    sys.modules["wanref.modules"] = modules_pkg
    sys.modules["wanref.modules.attention"] = attn
    sys.modules["wanref.modules.tokenizers"] = tok

    # torchvision is only touched via return_transforms=True, never in tests.
    # The stub needs a real ModuleSpec or diffusers' find_spec probe explodes.
    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv_t = types.ModuleType("torchvision.transforms")
        for name in ("Compose", "Resize", "ToTensor", "Normalize"):
            setattr(tv_t, name, object)
        tv_t.InterpolationMode = types.SimpleNamespace(BICUBIC="bicubic")
        tv.transforms = tv_t
        tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)
        tv_t.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", None)
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = tv_t


def _load_upstream(name: str):
    _ensure_wanref()
    full = f"wanref.modules.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, UPSTREAM_MODULES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def pt_scail2():
    return _load_upstream("model_scail2")


@pytest.fixture(scope="session")
def pt_clip():
    _load_upstream("xlm_roberta")
    return _load_upstream("clip")
