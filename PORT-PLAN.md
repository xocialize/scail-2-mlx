# SCAIL-2 → MLX port plan

Reference: `refs/SCAIL-2` (branch `wan-scail2`). Oracle = upstream PT code on
CPU fp32 with SDPA standing in for flash-attn. Workflow per the mlx-porting
skill: translate → parity-lock → pipeline → convert → e2e → quantize.

## Gates

### G1 — DiT translation + parity ✅ (2026-06-11)
- `modules/model_scail2.py` 1:1 translation (same names, decomposition, call
  order; complex RoPE → cos/sin fp32; convs NDHWC at call sites).
- RoPE parity: ref/video/pose segments + full 3-segment apply, both shift
  regimes — **max_abs < 1e-4** (8 tests).
- Full forward parity, tiny i2v config (dim 64, 2L), replace × history grid —
  **bit-exact (max_abs = 0.0)** fp32 (4 tests).
- 28-ch mask compression parity — exact (2 tests).
- State-dict key remap locked by strict loader in the forward tests.

### G2 — CLIP xlm-roberta ViT-H/14 visual tower ✅ (2026-06-11)
Real-weights parity fp32: max_abs 2.7e-4 / mean 1.7e-6. fp16 storage matches
upstream production `clip_dtype`. Discovered en route: M5 Max Metal fp32 GEMM
runs at ~8e-4 relative (TF32-class) — parity suite pinned to CPU stream.

### G3 — Weight conversion ✅ (2026-06-11)
dit 32.8 GB bf16 (1307/1307 strict keys), vae fp32, umt5 bf16, clip fp16.

### G4 — Pipeline ✅ (2026-06-11)
First e2e: animation_001 (end-to-end driving mode) 512×288 / 8 steps —
clean motion transfer, checkerboard detector clean, peak 102.6 GB,
~115 s/step (2× sequential CFG forwards, fp32 activations).

### G5 — open items
- 68-vs-65 frame decode count: mlx-video VAE temporal head emits 4·T frames
  for T latents; upstream emits 1+(T−1)·4. Find the 3 extra frames, align.
- bf16 activation pass + batched/collapsed CFG (2× speedup candidates).
- Golden vs PT: upstream requires CUDA flash-attn; CPU-oracle full-denoise
  comparison is hours-scale — decide scope (component parity + visual may
  suffice given results).

### ~~G2~~ (original plan below)
### G2 — CLIP xlm-roberta ViT-H/14 visual tower
- Port `refs/SCAIL-2/wan/modules/clip.py` visual branch (+ `xlm_roberta.py` is
  text-only — NOT needed: checkpoint is `onlyvisual`).
- Oracle: upstream CLIP on CPU with the released 2.5 GB checkpoint.
- Output: `clip.visual([img])` → [1, 257, 1280] tokens, parity < 1e-3 fp16.

### G3 — Weight conversion
- Download `zai-org/SCAIL-2` (81 GB). Run upstream `convert.py` → flat
  safetensors with Wan-style keys (CPU-only, splits fused QKV/KV).
- `recipes/convert_scail2.py`: per-component split (dit / vae / umt5 / clip),
  Conv3d → (O,*K,I), bf16 cast, **`mx.eval` before every save**.
- VAE + umT5: reuse mlx-video's existing converters/loaders where the key
  layout matches its Wan2.1 support.

### G4 — Pipeline (`scail.py`)
- 1:1 port of `SCAIL2Pipeline.generate`: segment builder (81/5 overlap),
  ref/pose VAE encode, 28-ch mask latents, history mask, CFG loop,
  `FlowUniPCScheduler` (mlx-video) with shift=5.0 (3.0 at 480p), 40 steps.
- `mx.eval` per denoise step (Metal command-buffer ceiling); model offload
  not needed (unified memory) but keep umT5/CLIP eviction hooks.
- Bilinear 0.5× resize for pose/driving-mask inputs — hand-rolled NHWC
  bilinear per skill `spatial-and-rope-ops.md` (align_corners=False).

### G5 — E2E golden + production dtype
- `examples/animation_001` (ref.jpg + rendered_v2.mp4 + masks) vs PT golden.
- bf16 production pass: norms/modulation stay fp32 (mirrors upstream
  autocast(float32) regions). Noise-path smoke + checkerboard detector.
- Wire 14B at 512p first; 704p second (~95k-token sequences — consider
  longcat-video-mlx BSA Metal kernels if SDPA is the wall).

### G6 — Quantize + publish
- q8/q4 transformer Linears (`group_size` 64), keep patch embeds / head /
  time-embed hi-precision; gate on per-pass cosine (int4 ≥ 0.99).
- License check before publish: GitHub says Apache-2.0, HF card says MIT.

## Notes / risks
- `SCAIL2Model` asserts `clip_fea is not None` — CLIP tower is a hard dep (G2
  before G4 e2e).
- Pose RoPE avg-pool produces non-unit rotations (|f| < 1) — intentional,
  do not normalize.
- Preprocessing (SCAIL-Pose: NLF/MMPose/SAM masks) stays upstream Python for
  now; pipeline consumes its rendered_v2.mp4 / mask videos directly.
- 1.3B config exists upstream (`config-1.3b.json`) but no released weights —
  ignore unless zai-org ships them.
