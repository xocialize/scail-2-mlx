"""Quantization smoke at tiny config: blocks-only scope, save/load roundtrip,
forward sanity (cosine vs full precision)."""
import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from scail2_mlx.modules.model_scail2 import SCAIL2Model
from scail2_mlx.utils.quant import blocks_linear_predicate, quantize_dit

mx.set_default_device(mx.cpu)

TINY = dict(
    model_type="i2v", patch_size=(1, 2, 2), text_len=512, in_dim=20, mask_dim=28,
    dim=64, ffn_dim=128, freq_dim=32, text_dim=48, out_dim=16, num_heads=4,
    num_layers=2, qk_norm=True, cross_attn_norm=True,
)


def _forward(model, seed=11):
    rng = np.random.default_rng(seed)
    g = lambda *s: mx.array(rng.standard_normal(s).astype("float32"))
    out = model(
        x=[g(16, 2, 8, 8)],
        pose_latents=[g(16, 2, 4, 4)],
        driving_masks=[g(28, 2, 4, 4)],
        ref_latents=[g(16, 1, 8, 8)],
        ref_masks=[g(28, 3, 8, 8)],
        t=mx.array([500.0]),
        context=[g(7, 48)],
        seq_len=int(1e10),
        replace_flag=False,
        clip_fea=g(1, 257, 1280),
    )[0]
    mx.eval(out)
    return np.array(out)


def test_quantize_scope_and_roundtrip(tmp_path):
    mx.random.seed(3)
    model = SCAIL2Model(**TINY)
    mx.eval(model.parameters())
    ref = _forward(model)

    quantize_dit(model, bits=8, group_size=32)

    # scope: block Linears quantized, embeddings/head untouched
    quant_paths = [
        p for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)
    ]
    assert quant_paths and all(p.startswith("blocks.") for p in quant_paths)
    assert not any("text_embedding" in p or "img_emb" in p or "head" in p
                   for p in quant_paths)

    q_out = _forward(model)
    cos = float(np.dot(ref.ravel(), q_out.ravel())
                / (np.linalg.norm(ref) * np.linalg.norm(q_out) + 1e-12))
    assert cos > 0.999, f"int8 cosine too low: {cos}"
    assert np.isfinite(q_out).all()

    # save/load roundtrip into a freshly-quantized skeleton
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(*weights.values())
    mx.save_safetensors(str(tmp_path / "dit-q8.safetensors"), weights)

    mx.random.seed(99)
    model2 = SCAIL2Model(**TINY)
    quantize_dit(model2, bits=8, group_size=32)
    model2.update(tree_unflatten(list(mx.load(str(tmp_path / "dit-q8.safetensors")).items())))
    mx.eval(model2.parameters())
    q_out2 = _forward(model2)
    assert np.abs(q_out - q_out2).max() < 1e-6
