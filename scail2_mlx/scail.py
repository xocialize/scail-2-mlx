# MLX port of refs/SCAIL-2/wan/scail.py SCAIL2Pipeline. Same generate()
# surface and segment/CFG/history flow; distributed/offload machinery dropped
# (unified memory). Substrate: mlx-video wan_2 (WanVAE, T5Encoder, FlowUniPC).
import logging
import math
import random
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from mlx_video.models.wan_2.scheduler import (
    FlowDPMPP2MScheduler,
    FlowUniPCScheduler,
)
from mlx_video.models.wan_2.text_encoder import T5Encoder
from mlx_video.models.wan_2.utils import encode_text
from mlx_video.models.wan_2.vae import WanVAE as _MlxVideoWanVAE

from .modules.clip import CLIPModel, clip_xlm_roberta_vit_h_14
from .modules.model_scail2 import SCAIL2Model
from .utils.scail_utils import extract_and_compress_mask_to_latent
from .utils.vae_stream import decode_chunked

__all__ = ["SCAIL2Pipeline"]


def _load_component(model, path):
    weights = mx.load(str(path))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def _half_bilinear(x):
    # F.interpolate(scale_factor=0.5, mode='bilinear', align_corners=False)
    # at an exact 1/2 ratio reduces to 2x2 mean pooling
    *lead, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0
    x = x.reshape(*lead, h // 2, 2, w // 2, 2)
    return x.mean(axis=(-3, -1))


class _WanVAEAdapter:
    """Upstream-API shim (list-of-[C,T,H,W] in/out) over mlx-video WanVAE."""

    def __init__(self, weights_path):
        self.model = _MlxVideoWanVAE(z_dim=16, encoder=True)
        _load_component(self.model, weights_path)

    def encode(self, videos):
        return [self.model.encode(u[None].astype(mx.float32))[0] for u in videos]

    def decode(self, zs):
        # decode_chunked, not model.decode: upstream-faithful 1+(T-1)*4 frame
        # count and first-chunk handling (whole-seq decode emits 4*T frames
        # with a divergent head and +3-frame phase shift)
        return [decode_chunked(self.model, z[None].astype(mx.float32))[0] for z in zs]


class _T5EncoderAdapter:
    """Upstream-API shim: callable([prompts]) -> list of trimmed [L, C]."""

    def __init__(self, weights_path, tokenizer_path, text_len):
        from transformers import AutoTokenizer

        self.model = T5Encoder()
        _load_component(self.model, weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.text_len = text_len

    def __call__(self, prompts):
        return [
            encode_text(self.model, self.tokenizer, p, self.text_len)
            for p in prompts
        ]


class SCAIL2Pipeline:

    def __init__(
        self,
        config,
        weights_dir,
        tokenizer_dir=None,
        lora_path=None,
        lora_alpha=None,
        evict_encoders=True,
    ):
        r"""
        Args:
            config: SimpleNamespace from scail2_mlx.configs (scail_14B)
            weights_dir: directory holding converted MLX safetensors
                (dit.safetensors, vae.safetensors, umt5.safetensors,
                clip.safetensors) — output of recipes/convert_scail2.py
            tokenizer_dir: umt5-xxl tokenizer directory (defaults to the
                bundled weights/SCAIL-2/umt5-xxl)
        """
        from pathlib import Path

        weights_dir = Path(weights_dir)
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.text_encoder = _T5EncoderAdapter(
            weights_dir / "umt5.safetensors",
            tokenizer_dir or "google/umt5-xxl",
            config.text_len,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = _WanVAEAdapter(weights_dir / "vae.safetensors")

        self.clip = CLIPModel(
            model=_load_component(
                clip_xlm_roberta_vit_h_14(), weights_dir / "clip.safetensors"
            ),
            dtype=config.clip_dtype,
        )

        logging.info(f"Creating SCAIL2Model from {weights_dir / 'dit.safetensors'}")
        self.model = SCAIL2Model(
            model_type=config.model_type,
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=config.in_dim,
            mask_dim=config.mask_dim,
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            out_dim=config.out_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
        )
        _load_component(self.model, weights_dir / "dit.safetensors")
        if lora_path is not None:
            raise NotImplementedError("LoRA fusing lands with G6")

        self.sample_neg_prompt = config.sample_neg_prompt
        # umT5 (11 GB) + CLIP (1.2 GB) are needed once per generate() call;
        # dropping them after encoding cuts the denoise-phase peak (lance-mlx
        # PR #6 relay pattern, trivial tier). Makes the pipeline single-shot.
        self.evict_encoders = evict_encoders

    def generate(
        self,
        input_prompt,
        img,
        ref_mask_img,
        pose_video,
        driving_mask_video,
        replace_flag: bool,
        segment_len=81,
        segment_overlap=5,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt=None,
        seed=-1,
        noise_override=None,
        batched_cfg=True,
    ):
        r"""
        Generates video from a reference image + driving inputs. Argument
        semantics identical to upstream SCAIL2Pipeline.generate:

            img:                [3, H, W] in [-1, 1]
            ref_mask_img:       [3, H, W] in [-1, 1]
            pose_video:         [T, 3, H, W] in [-1, 1]
            driving_mask_video: [3, T, H, W] in [-1, 1]

        `noise_override` is a test-only hook: a callable (shape) -> mx.array
        injected in place of seeded mx.random noise, so golden comparisons
        can share numpy noise with the PT reference (RNGs are not
        cross-compatible).

        Returns [3, T_out, H_out, W_out] in [-1, 1].
        """
        if segment_len <= 0:
            raise ValueError("segment_len must be positive")
        if segment_overlap <= 0 or segment_overlap >= segment_len:
            raise ValueError("segment_overlap must be in (0, segment_len)")

        img = mx.array(img) if not isinstance(img, mx.array) else img
        ref_mask_img = (
            mx.array(ref_mask_img)
            if not isinstance(ref_mask_img, mx.array)
            else ref_mask_img
        )
        pose_video = (
            mx.array(pose_video) if not isinstance(pose_video, mx.array) else pose_video
        )
        driving_mask_video = (
            mx.array(driving_mask_video)
            if not isinstance(driving_mask_video, mx.array)
            else driving_mask_video
        )
        ori_img = img[None]  # 1, 3, H, W

        num_frames = pose_video.shape[0]
        if driving_mask_video.shape[1] != num_frames:
            raise ValueError(
                f"pose_video and driving_mask_video must have the same frame count, "
                f"got {num_frames} and {driving_mask_video.shape[1]}"
            )

        def build_segments(total_frames):
            if total_frames <= segment_len:
                keep = ((total_frames - 1) // self.vae_stride[0]) * self.vae_stride[0] + 1
                return [(0, keep)]
            segments = []
            start = 0
            stride = segment_len - segment_overlap
            while start < total_frames:
                end = start + segment_len
                if end > total_frames:
                    break
                segments.append((start, end))
                start += stride
            return segments

        segments = build_segments(num_frames)
        if len(segments) == 0:
            raise ValueError(
                f"No valid segment was produced for {num_frames} frames. "
                f"Use a longer driving video or reduce segment_len."
            )
        if len(segments) > 1:
            logging.info(
                f"Sampling {len(segments)} segments with segment_len={segment_len}, "
                f"segment_overlap={segment_overlap}."
            )

        ref_latent = self.vae.encode([ori_img.transpose(1, 0, 2, 3)])[0]
        ref_mask_latent_28ch = extract_and_compress_mask_to_latent(
            ref_mask_img[:, None], additional_spatial_downsample=1
        )  # (28, 1, H_lat, W_lat)
        lat_c = ref_latent.shape[0]

        seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        mx.random.seed(seed)

        if n_prompt is None:
            n_prompt = ""

        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])

        clip_context = self.clip.visual([img[:, None, :, :]])
        mx.eval(clip_context, *context, *context_null)

        if self.evict_encoders:
            self.text_encoder = None
            self.clip = None
            mx.clear_cache()

        # cap the Metal buffer cache: without a limit, per-step transient
        # buffers accumulate as cached-but-unused memory over a 40-step loop
        mx.set_cache_limit(8 * 1024**3)

        def apply_clean_history(latent, history_latent):
            if history_latent is None:
                return latent
            history_t = history_latent.shape[1]
            return mx.concatenate(
                [history_latent.astype(latent.dtype), latent[:, history_t:]], axis=1
            )

        output_segments = []
        prev_history_pixel = None

        def build_sample_scheduler():
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCScheduler(
                    num_train_timesteps=self.num_train_timesteps
                )
                sample_scheduler.set_timesteps(sampling_steps, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMPP2MScheduler(
                    num_train_timesteps=self.num_train_timesteps
                )
                sample_scheduler.set_timesteps(sampling_steps, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")
            return sample_scheduler, timesteps

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            logging.info(
                f"Processing segment {seg_idx + 1}/{len(segments)}: "
                f"frames [{seg_start}, {seg_end})"
            )
            sample_scheduler, timesteps = build_sample_scheduler()

            pose_segment = pose_video[seg_start:seg_end]
            smpl_render_video = _half_bilinear(pose_segment)
            pose_latent = self.vae.encode(
                [smpl_render_video.transpose(1, 0, 2, 3)]
            )[0]

            lat_t = pose_latent.shape[1]
            _, lat_h, lat_w = ref_latent.shape[1:]

            null_noisy_mask = mx.zeros(
                (ref_mask_latent_28ch.shape[0], lat_t, lat_h, lat_w),
                dtype=ref_mask_latent_28ch.dtype,
            )
            ref_masks = mx.concatenate([ref_mask_latent_28ch, null_noisy_mask], axis=1)

            driving_mask_segment = driving_mask_video[:, seg_start:seg_end]
            driving_mask_segment = _half_bilinear(driving_mask_segment)
            driving_masks = extract_and_compress_mask_to_latent(
                driving_mask_segment, additional_spatial_downsample=1
            )

            history_latent = None
            history_mask = None
            if seg_idx > 0:
                if prev_history_pixel is None:
                    raise RuntimeError("Missing previous segment history frames.")
                history_latent = self.vae.encode([prev_history_pixel])[0]
                history_t = min(history_latent.shape[1], lat_t)
                history_mask = mx.concatenate(
                    [
                        mx.ones((4, history_t, lat_h, lat_w), dtype=mx.float32),
                        mx.zeros((4, lat_t - history_t, lat_h, lat_w), dtype=mx.float32),
                    ],
                    axis=1,
                )
                logging.info(
                    f"Using {prev_history_pixel.shape[1]} clean history frames "
                    f"({history_t} latent frames)."
                )

            if noise_override is not None:
                noise = noise_override((lat_c, lat_t, lat_h, lat_w))
            else:
                noise = mx.random.normal((lat_c, lat_t, lat_h, lat_w), dtype=mx.float32)

            arg_c = {
                "context": [context[0]],
                "clip_fea": clip_context,
                "seq_len": int(1e10),
                "ref_latents": [ref_latent],
                "ref_masks": [ref_masks],
                "pose_latents": [pose_latent],
                "driving_masks": [driving_masks],
                "history_mask": [history_mask] if history_mask is not None else None,
                "replace_flag": replace_flag,
            }

            arg_null = {
                "context": context_null,
                "clip_fea": clip_context,
                "seq_len": int(1e10),
                "ref_latents": [ref_latent],
                "ref_masks": [ref_masks],
                "pose_latents": [pose_latent],
                "driving_masks": [driving_masks],
                "history_mask": [history_mask] if history_mask is not None else None,
                "replace_flag": replace_flag,
            }

            # batched CFG: one B=2 forward [cond, uncond] instead of two
            # sequential B=1 forwards — numerically identical, better GPU
            # occupancy. Falls back to sequential when batched_cfg=False or
            # guidance is off.
            use_batched = batched_cfg and guide_scale > 1.0
            if use_batched:
                arg_b = {
                    "context": [arg_c["context"][0], arg_null["context"][0]],
                    "clip_fea": mx.concatenate([clip_context, clip_context], axis=0),
                    "seq_len": int(1e10),
                    "ref_latents": [ref_latent, ref_latent],
                    "ref_masks": [ref_masks, ref_masks],
                    "pose_latents": [pose_latent, pose_latent],
                    "driving_masks": [driving_masks, driving_masks],
                    "history_mask": (
                        [history_mask, history_mask]
                        if history_mask is not None
                        else None
                    ),
                    "replace_flag": replace_flag,
                }

            latent = apply_clean_history(noise, history_latent)
            for step_idx, t in enumerate(timesteps):
                model_input = apply_clean_history(latent, history_latent)

                if use_batched:
                    timestep = mx.array([t, t])
                    preds = self.model(
                        [model_input, model_input], t=timestep, **arg_b
                    )
                    noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    timestep = mx.array([t])
                    noise_pred_cond = self.model(
                        [model_input], t=timestep, **arg_c
                    )[0]
                    if guide_scale <= 1.0:
                        noise_pred = noise_pred_cond
                    else:
                        noise_pred_uncond = self.model(
                            [model_input], t=timestep, **arg_null
                        )[0]
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )

                temp_x0 = sample_scheduler.step(
                    noise_pred[None], t, latent[None]
                )[0]
                latent = apply_clean_history(temp_x0, history_latent)
                # bound the lazy graph per denoise step (Metal command-buffer
                # timeout) and keep peak memory flat; clear the Metal buffer
                # cache so freed step workspace doesn't ratchet RSS upward
                # across the loop (the silent-SIGKILL failure mode)
                mx.eval(latent)
                mx.clear_cache()
                logging.info(
                    f"  step {step_idx + 1}/{len(timesteps)} "
                    f"active={mx.get_active_memory() / 1e9:.1f}GB "
                    f"peak={mx.get_peak_memory() / 1e9:.1f}GB"
                )

            videos = self.vae.decode([latent])
            segment_video = videos[0]
            mx.eval(segment_video)

            if seg_idx == 0:
                output_segments.append(segment_video)
            else:
                output_segments.append(segment_video[:, segment_overlap:])
            if seg_idx < len(segments) - 1:
                prev_history_pixel = segment_video[:, -segment_overlap:]

        return mx.concatenate(output_segments, axis=1)
