"""Load the upstream PT reference (refs/SCAIL-2) importable on CPU.

Upstream `wan.modules.attention.flash_attention` asserts CUDA, and the `wan`
package __init__ drags in decord/distributed. We load model_scail2.py alone
under a synthetic package, with a CPU SDPA stand-in pre-injected for its
`.attention` relative import.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "refs" / "SCAIL-2" / "wan" / "modules" / "model_scail2.py"


def _cpu_flash_attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None,
    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
    dtype=None, version=None,
):
    import torch.nn.functional as F

    assert tuple(window_size) == (-1, -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


@pytest.fixture(scope="session")
def pt_scail2():
    pkg = types.ModuleType("wanref")
    pkg.__path__ = []
    modules_pkg = types.ModuleType("wanref.modules")
    modules_pkg.__path__ = []
    attn = types.ModuleType("wanref.modules.attention")
    attn.flash_attention = _cpu_flash_attention
    sys.modules["wanref"] = pkg
    sys.modules["wanref.modules"] = modules_pkg
    sys.modules["wanref.modules.attention"] = attn

    spec = importlib.util.spec_from_file_location(
        "wanref.modules.model_scail2", UPSTREAM
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wanref.modules.model_scail2"] = mod
    spec.loader.exec_module(mod)
    return mod
