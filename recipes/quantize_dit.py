"""Quantize the SCAIL-2 DiT to q8/q4 (transformer-block Linears only).

Produces dit-q{bits}.safetensors + quant.json in --out (which the pipeline
auto-detects). --verify runs a per-pass cosine gate of quantized vs bf16 on
identical injected inputs (skill thresholds: int4 >= 0.99, int8 >= 0.9999 —
NOT image-level PSNR; quantization legitimately perturbs the trajectory).

Usage:
  .venv/bin/python recipes/quantize_dit.py --bits 4 [--verify]
  .venv/bin/python recipes/quantize_dit.py --bits 8 --group-size 128 [--verify]
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scail2_mlx.configs import scail_14B as cfg  # noqa: E402
from scail2_mlx.modules.model_scail2 import SCAIL2Model  # noqa: E402
from scail2_mlx.utils.quant import quantize_dit, write_quant_config  # noqa: E402


def _build_model():
    return SCAIL2Model(
        model_type=cfg.model_type, patch_size=cfg.patch_size, text_len=cfg.text_len,
        in_dim=cfg.in_dim, mask_dim=cfg.mask_dim, dim=cfg.dim, ffn_dim=cfg.ffn_dim,
        freq_dim=cfg.freq_dim, out_dim=cfg.out_dim, num_heads=cfg.num_heads,
        num_layers=cfg.num_layers, window_size=cfg.window_size, qk_norm=cfg.qk_norm,
        cross_attn_norm=cfg.cross_attn_norm, eps=cfg.eps,
    )


def _load(model, path):
    model.update(tree_unflatten(list(mx.load(str(path)).items())))
    mx.eval(model.parameters())
    return model


def _forward(model, rng):
    T, H, W = 2, 16, 16
    g = lambda *s: mx.array(rng.standard_normal(s).astype("float32"))
    out = model(
        x=[g(16, T, H, W)],
        pose_latents=[g(16, T, H // 2, W // 2)],
        driving_masks=[g(28, T, H // 2, W // 2)],
        ref_latents=[g(16, 1, H, W)],
        ref_masks=[g(28, 1 + T, H, W)],
        t=mx.array([500.0]),
        context=[g(20, 4096)],
        seq_len=int(1e10),
        replace_flag=False,
        clip_fea=g(1, 257, 1280),
    )[0]
    mx.eval(out)
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="weights/mlx/dit.safetensors")
    p.add_argument("--out", default=None, help="default: weights/mlx-q{bits}")
    p.add_argument("--bits", type=int, choices=[4, 8], required=True)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out or f"weights/mlx-q{args.bits}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dit_file = f"dit-q{args.bits}.safetensors"

    print(f"loading bf16 DiT from {args.src}...")
    model = _load(_build_model(), args.src)

    ref_out = None
    rng_seed = 1234
    if args.verify:
        print("bf16 reference forward...")
        ref_out = _forward(model, np.random.default_rng(rng_seed))

    print(f"quantizing: {args.bits}-bit, group_size {args.group_size}, blocks-only...")
    quantize_dit(model, bits=args.bits, group_size=args.group_size)
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(*weights.values())
    mx.save_safetensors(str(out_dir / dit_file), weights)
    write_quant_config(out_dir, args.bits, args.group_size, dit_file)
    total = sum(v.size * v.dtype.size for v in weights.values()) / 1e9
    print(f"wrote {out_dir / dit_file} ({total:.2f} GB) + quant.json")

    # make the q-dir self-contained: link the unquantized components
    src_dir = Path(args.src).parent
    for comp in ("vae.safetensors", "umt5.safetensors", "clip.safetensors"):
        link = out_dir / comp
        if not link.exists() and (src_dir / comp).exists():
            link.symlink_to((src_dir / comp).resolve())
            print(f"linked {comp}")

    if args.verify:
        print("quantized forward (identical inputs)...")
        q_out = _forward(model, np.random.default_rng(rng_seed))
        cos = float(
            np.dot(ref_out.ravel(), q_out.ravel())
            / (np.linalg.norm(ref_out) * np.linalg.norm(q_out) + 1e-12)
        )
        gate = 0.99 if args.bits == 4 else 0.9999
        print(f"per-pass cosine vs bf16: {cos:.6f} (gate {gate})")
        print("PASS" if cos >= gate else "FAIL — do not ship this quant")


if __name__ == "__main__":
    main()
