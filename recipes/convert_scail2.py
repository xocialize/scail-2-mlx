"""Convert the zai-org/SCAIL-2 HF checkpoint to MLX safetensors.

Components (each independently loadable, selectable via --components):
  dit   model/1/fsdp2_rank_0000_checkpoint.pt (fp32, SAT key space)
        -> upstream refs/SCAIL-2/convert.py key remap (splits fused QKV/KV)
        -> MLX key remap (Sequential .layers., Conv3d NDHWC) -> bf16
  vae   Wan2.1_VAE.pth         -> mlx-video sanitize -> fp32
  umt5  umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth -> mlx-video sanitize -> bf16
  clip  models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
        -> visual.* + log_scale only -> MLX remap + Conv2d NDHWC -> fp16

Every tensor is mx.eval'd before save — lazy tensors serialize as zeros.

Usage:
  .venv/bin/python recipes/convert_scail2.py \
      --src weights/SCAIL-2 --out weights/mlx [--components dit,vae,umt5,clip]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "mlx-video"))

from scail2_mlx.utils.weights import is_conv_weight, pt_key_to_mlx, _transpose_conv  # noqa: E402


def _load_upstream_convert():
    spec = importlib.util.spec_from_file_location(
        "scail_upstream_convert", REPO_ROOT / "refs" / "SCAIL-2" / "convert.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(out_path: Path, weights: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(*weights.values())
    mx.save_safetensors(str(out_path), weights)
    total = sum(v.size * v.dtype.size for v in weights.values()) / 1e9
    print(f"  wrote {out_path} ({len(weights)} tensors, {total:.2f} GB)")


def convert_dit(src: Path, out: Path):
    import torch

    upstream = _load_upstream_convert()
    ckpt_path = src / "model" / "1" / "fsdp2_rank_0000_checkpoint.pt"
    print(f"loading {ckpt_path} (this allocates ~56 GB)...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["module"]
    del checkpoint

    print("remapping SAT keys -> Wan keys (upstream convert.py)...")
    wan_state = upstream.get_new_state_dict(state_dict)
    del state_dict

    print("remapping Wan keys -> MLX, casting bf16...")
    weights = {}
    for key in sorted(wan_state.keys()):
        arr = wan_state.pop(key).detach().float().numpy()
        if is_conv_weight(key):
            arr = _transpose_conv(np.ascontiguousarray(arr))
        weights[pt_key_to_mlx(key)] = mx.array(arr).astype(mx.bfloat16)
    _save(out / "dit.safetensors", weights)


def convert_vae(src: Path, out: Path):
    from mlx_video.models.wan_2.convert import (
        load_torch_weights,
        sanitize_wan_vae_weights,
    )

    weights = load_torch_weights(str(src / "Wan2.1_VAE.pth"))
    weights = sanitize_wan_vae_weights(weights)
    _save(out / "vae.safetensors", weights)


def convert_umt5(src: Path, out: Path):
    from mlx_video.models.wan_2.convert import (
        load_torch_weights,
        sanitize_wan_t5_weights,
    )

    weights = load_torch_weights(
        str(src / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth")
    )
    weights = sanitize_wan_t5_weights(weights)
    weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
    _save(out / "umt5.safetensors", weights)


def convert_clip(src: Path, out: Path):
    import torch

    pth = src / "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth"
    state = torch.load(str(pth), map_location="cpu", weights_only=True)
    weights = {}
    dropped = []
    for key, value in state.items():
        if not (key.startswith("visual.") or key == "log_scale"):
            dropped.append(key)
            continue
        arr = value.detach().float().numpy()
        if is_conv_weight(key):
            arr = _transpose_conv(np.ascontiguousarray(arr))
        weights[pt_key_to_mlx(key)] = mx.array(arr).astype(mx.float16)
    if dropped:
        print(f"  dropped non-visual keys: {sorted(dropped)[:5]} (+{len(dropped)-5 if len(dropped)>5 else 0})")
    _save(out / "clip.safetensors", weights)


CONVERTERS = {
    "dit": convert_dit,
    "vae": convert_vae,
    "umt5": convert_umt5,
    "clip": convert_clip,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="weights/SCAIL-2")
    p.add_argument("--out", default="weights/mlx")
    p.add_argument("--components", default="vae,umt5,clip,dit")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    for name in args.components.split(","):
        print(f"== converting {name} ==")
        CONVERTERS[name.strip()](src, out)


if __name__ == "__main__":
    main()
