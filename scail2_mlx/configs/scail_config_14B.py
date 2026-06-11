# Mirror of refs/SCAIL-2/wan/configs/scail_config_14B.py + shared_config.py
# (EasyDict -> SimpleNamespace; torch dtypes -> mlx dtypes).
from types import SimpleNamespace

import mlx.core as mx

scail_14B = SimpleNamespace(
    __name__="Config: SCAIL 14B",
    # shared
    t5_dtype=mx.bfloat16,
    text_len=512,
    param_dtype=mx.bfloat16,
    num_train_timesteps=1000,
    sample_fps=16,
    sample_neg_prompt="",
    # t5
    t5_checkpoint="umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth",
    t5_tokenizer="umt5-xxl",
    # clip
    clip_model="clip_xlm_roberta_vit_h_14",
    clip_dtype=mx.float16,
    clip_checkpoint="models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth",
    clip_tokenizer="xlm-roberta-large",
    # vae
    vae_checkpoint="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    # transformer (config-14b.json: in_dim 20, mask_dim 28, model_type i2v)
    model_type="i2v",
    patch_size=(1, 2, 2),
    in_dim=20,
    mask_dim=28,
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    out_dim=16,
    num_heads=40,
    num_layers=40,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)
