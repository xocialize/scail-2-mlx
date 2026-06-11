"""SCAIL-2 MLX generation CLI — mirrors refs/SCAIL-2/generate.py input prep.

Example (animation mode, bundled upstream example):
  .venv/bin/python scripts/generate.py \
      --weights-dir weights/mlx \
      --tokenizer-dir weights/SCAIL-2/umt5-xxl \
      --image refs/SCAIL-2/examples/animation_001/ref.jpg \
      --mask-image refs/SCAIL-2/examples/animation_001/ref_mask.jpg \
      --pose refs/SCAIL-2/examples/animation_001/rendered_v2.mp4 \
      --mask-video refs/SCAIL-2/examples/animation_001/rendered_mask_v2.mp4 \
      --prompt "..." --target-h 704 --target-w 384 --save-file out.mp4

Image/video resizing uses PIL bicubic (antialiased), which is not bit-equal to
upstream's torchvision resize — golden parity tests bypass this script and
inject preprocessed tensors directly into SCAIL2Pipeline.generate().
"""
import argparse
import logging
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "refs" / "mlx-video"))

import mlx.core as mx  # noqa: E402

from scail2_mlx.configs import scail_14B  # noqa: E402
from scail2_mlx.scail import SCAIL2Pipeline  # noqa: E402


def _resize_for_rectangle_crop_np(frames_thwc, target_h, target_w):
    """Upstream resize_for_rectangle_crop (reshape_mode='center'), PIL bicubic."""
    t, h, w, c = frames_thwc.shape
    if w / h > target_w / target_h:
        new_h, new_w = target_h, round(w * target_h / h)
    else:
        new_h, new_w = round(h * target_w / w), target_w
    out = np.empty((t, new_h, new_w, c), dtype=np.uint8)
    for i in range(t):
        out[i] = np.asarray(
            Image.fromarray(frames_thwc[i]).resize((new_w, new_h), Image.BICUBIC)
        )
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return out[:, top : top + target_h, left : left + target_w]


def _load_image(path, target_h, target_w):
    img = np.asarray(Image.open(path).convert("RGB"))[None]  # 1 h w c
    img = _resize_for_rectangle_crop_np(img, target_h, target_w)[0]
    return (img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 127.5  # c h w


def _load_video(path, target_h, target_w, max_frames=None):
    frames = iio.imread(path, plugin="pyav")  # t h w c uint8
    if max_frames:
        frames = frames[:max_frames]
    frames = _resize_for_rectangle_crop_np(frames, target_h, target_w)
    return (frames.astype(np.float32).transpose(0, 3, 1, 2) - 127.5) / 127.5  # t c h w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", default="weights/mlx")
    p.add_argument("--tokenizer-dir", default="weights/SCAIL-2/umt5-xxl")
    p.add_argument("--image", required=True)
    p.add_argument("--mask-image", required=True)
    p.add_argument("--pose", required=True)
    p.add_argument("--mask-video", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--replace-flag", action="store_true")
    p.add_argument("--target-h", type=int, required=True, help="divisible by 32")
    p.add_argument("--target-w", type=int, required=True, help="divisible by 32")
    p.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--sample-shift", type=float, default=5.0)
    p.add_argument("--sample-guide-scale", type=float, default=5.0)
    p.add_argument("--segment-len", type=int, default=81)
    p.add_argument("--segment-overlap", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-file", default="scail2_out.mp4")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert args.target_h % 32 == 0 and args.target_w % 32 == 0

    img = _load_image(args.image, args.target_h, args.target_w)
    mask_img = _load_image(args.mask_image, args.target_h, args.target_w)
    pose_video = _load_video(args.pose, args.target_h, args.target_w, args.max_frames)
    mask_video = _load_video(
        args.mask_video, args.target_h, args.target_w, args.max_frames
    ).transpose(1, 0, 2, 3)  # t c h w -> c t h w

    logging.info(
        f"inputs: img {img.shape}, pose {pose_video.shape}, mask {mask_video.shape}"
    )

    pipeline = SCAIL2Pipeline(
        scail_14B, args.weights_dir, tokenizer_dir=args.tokenizer_dir
    )

    video = pipeline.generate(
        args.prompt,
        mx.array(img),
        ref_mask_img=mx.array(mask_img),
        pose_video=mx.array(pose_video),
        driving_mask_video=mx.array(mask_video),
        replace_flag=args.replace_flag,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        segment_len=args.segment_len,
        segment_overlap=args.segment_overlap,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
    )

    out = np.array(video)  # 3 t h w, [-1, 1]
    out = ((out.transpose(1, 2, 3, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    iio.imwrite(args.save_file, out, plugin="pyav", fps=scail_14B.sample_fps, codec="libx264")
    logging.info(f"saved {args.save_file} ({out.shape[0]} frames)")
    print(f"peak memory: {mx.get_peak_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
