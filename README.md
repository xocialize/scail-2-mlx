# scail-2-mlx

Apple MLX port of **[zai-org/SCAIL-2](https://github.com/zai-org/SCAIL-2)** —
end-to-end controlled character animation (reference character image + driving
video → animated video), CVPR 2026 Findings lineage, arXiv 2512.05905. Supports
cross-identity replacement, multi-character scenes, and animal driving without
intermediate pose representations.

> **Status: end-to-end generation working.** All components parity-locked vs
> the PT CPU oracle (22 tests; CLIP validated on real weights at max_abs
> 2.7e-4 fp32), full 14B checkpoint converted (1307/1307 keys), and the
> bundled `animation_001` example generates clean motion transfer on an
> M5 Max: reference identity/environment preserved, driving dance reproduced,
> zero checkerboard artifacts, at 8 sampling steps / 512×288 (~115 s/step,
> peak 103 GB, fp32 activations — bf16 + batched-CFG speed pass pending).
> See [PORT-PLAN.md](PORT-PLAN.md) for remaining gates (G5 golden/perf, G6
> quant + publish).

## Why this port is cheap

SCAIL-2 is a fork of the official Wan2.1 codebase. Everything that is not
SCAIL-specific rides the **mlx-video Wan2 substrate** (same reuse path as
`bernini-r-mlx` and `phantom-wan-mlx`): umT5-XXL encoder, Wan2.1 16-ch VAE,
`FlowUniPCScheduler`. The DiT itself is translated 1:1 in
`scail2_mlx/modules/model_scail2.py` because its RoPE and embedding surface is
the novelty.

## The SCAIL-2 delta (net-new surface)

1. **3-segment RoPE** — sequence is `[ref | video | pose]`; each segment gets
   its own (t, h, w) position shifts (120-offset scheme, different between
   animation and replacement modes). The pose segment's rotary frequencies are
   **avg-pooled 2× spatially as complex numbers** (cos/sin pooled separately,
   magnitude < 1 — intentional, replicated exactly).
2. **Dual mask patch embeddings** — `patch_embedding_pose` (20ch) and
   `patch_embedding_mask` (28ch color-coded binary masks, 7 colors × 4 packed
   frames).
3. **i2v cross-attention** — CLIP image tokens (`k_img`/`v_img`) added to text
   cross-attention (Wan2.1-I2V style; needs the open-clip xlm-roberta ViT-H/14
   visual tower).
4. **Segmented long-video generation** — 81-frame segments with 5-frame clean
   history overlap.

## Layout

```
scail2_mlx/
  configs/scail_config_14B.py   # mirror of upstream config (dim 5120, 40L/40H)
  modules/model_scail2.py       # SCAIL2 DiT, 1:1 translation  [parity-locked]
  modules/attention.py          # flash_attention shim -> mx.fast SDPA
  modules/clip.py               # CLIP xlm-roberta ViT-H visual  (next)
  scail.py                      # SCAIL2Pipeline                 (next)
  utils/scail_utils.py          # 28-ch mask compression         [parity-locked]
  utils/weights.py              # PT->MLX key remap + loader     [parity-locked]
recipes/                        # weight conversion              (next)
tests/parity/                   # PT-oracle tests (CPU, fp32)
refs/SCAIL-2/                   # upstream clone (gitignored)
refs/mlx-video/                 # Wan2 substrate (gitignored, editable install)
```

## Dev setup

```bash
uv venv --python 3.12 .venv
uv pip install -e refs/mlx-video -e ".[parity]"
.venv/bin/python -m pytest tests/parity -q
```

## Weights

`zai-org/SCAIL-2` on HF (81 GB): FSDP `.pt` DiT checkpoint + Wan2.1 VAE +
umT5-XXL + CLIP. Conversion path: upstream `convert.py` (CPU key remap, splits
fused QKV) → Wan-style safetensors → MLX recipe (Conv3d → NDHWC, per-component
split, `mx.eval` before save).

## License

Apache-2.0, following the upstream GitHub repo. (HF model card states MIT —
verify weight license before any mlx-community publish.) Derived from SCAIL-2
(Zhipu AI / zai-org) and Wan2.1 (Alibaba Wan team).
