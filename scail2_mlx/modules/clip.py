# MLX port of refs/SCAIL-2/wan/modules/clip.py — visual tower only.
# The SCAIL-2 checkpoint (models_clip_...-onlyvisual.pth) ships no text
# branch, and the pipeline only calls CLIPModel.visual(); XLMRoberta* text
# classes are intentionally not ported. Structure/names mirror upstream.
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import flash_attention

__all__ = [
    "XLMRobertaCLIP",
    "clip_xlm_roberta_vit_h_14",
    "CLIPModel",
]


def pos_interpolate(pos, seq_len):
    if pos.shape[1] == seq_len:
        return pos
    raise NotImplementedError(
        "pos interpolation not needed at fixed 224x224 input"
    )


class QuickGELU(nn.Module):

    def __call__(self, x):
        return x * mx.sigmoid(1.702 * x)


class LayerNorm(nn.Module):
    # fp32 internal compute, like upstream LayerNorm(x.float()).type_as(x)

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))

    def __call__(self, x):
        return mx.fast.layer_norm(
            x.astype(mx.float32), self.weight, self.bias, self.eps
        ).astype(x.dtype)


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, causal=False, attn_dropout=0.0, proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x):
        """
        x:   [B, L, C].
        """
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim

        # compute query, key, value
        qkv = self.to_qkv(x).reshape(b, s, 3, n, d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # compute attention (inference: dropout 0; ViT branch is non-causal)
        assert not self.causal
        x = flash_attention(q, k, v)
        x = x.reshape(b, s, c)

        # output
        x = self.proj(x)
        return x


class SwiGLU(nn.Module):

    def __init__(self, dim, mid_dim):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def __call__(self, x):
        x = nn.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        post_norm=False,
        causal=False,
        activation="quick_gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        norm_eps=1e-5,
    ):
        assert activation in ["quick_gelu", "gelu", "swi_glu"]
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout, proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == "swi_glu":
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == "quick_gelu" else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(proj_dropout),
            )

    def __call__(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        activation="gelu",
        proj_dropout=0.0,
        norm_eps=1e-5,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = gain * mx.random.normal((1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_dropout),
        )

    def __call__(self, x):
        """
        x:  [B, L, C].
        """
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim

        # compute query, key, value
        q = mx.broadcast_to(
            self.to_q(self.cls_embedding).reshape(1, 1, n, d), (b, 1, n, d)
        )
        kv = self.to_kv(x).reshape(b, s, 2, n, d)
        k, v = kv[:, :, 0], kv[:, :, 1]

        # compute attention
        x = flash_attention(q, k, v)
        x = x.reshape(b, 1, c)

        # output
        x = self.proj(x)

        # mlp
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        mlp_ratio=4,
        out_dim=512,
        num_heads=12,
        num_layers=12,
        pool_type="token",
        pre_norm=True,
        post_norm=False,
        activation="quick_gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
    ):
        assert image_size % patch_size == 0
        assert pool_type in ("token", "token_fc", "attn_pool")
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=not pre_norm
        )
        if pool_type in ("token", "token_fc"):
            self.cls_embedding = gain * mx.random.normal((1, 1, dim))
        self.pos_embedding = gain * mx.random.normal(
            (1, self.num_patches + (1 if pool_type in ("token", "token_fc") else 0), dim)
        )

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = [
            AttentionBlock(
                dim, mlp_ratio, num_heads, post_norm, False, activation,
                attn_dropout, proj_dropout, norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # head
        if pool_type == "token":
            self.head = gain * mx.random.normal((dim, out_dim))
        elif pool_type == "token_fc":
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == "attn_pool":
            self.head = AttentionPool(
                dim, mlp_ratio, num_heads, activation, proj_dropout, norm_eps
            )

    def __call__(self, x, interpolation=False, use_31_block=False):
        """
        x: [B, 3, H, W] (NCHW, mirroring upstream interface)
        """
        b = x.shape[0]

        # embeddings (conv runs NHWC)
        x = self.patch_embedding(x.transpose(0, 2, 3, 1))
        x = x.reshape(b, -1, self.dim)  # NHWC -> [B, h*w, C], matches flatten(2).permute
        if self.pool_type in ("token", "token_fc"):
            x = mx.concatenate(
                [mx.broadcast_to(self.cls_embedding, (b, 1, self.dim)), x], axis=1
            )
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.shape[1])
        else:
            e = self.pos_embedding
        x = x + e
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            for block in self.transformer[:-1]:
                x = block(x)
            return x
        else:
            for block in self.transformer:
                x = block(x)
            return x


class XLMRobertaCLIP(nn.Module):
    # visual branch only (the SCAIL-2 checkpoint is "onlyvisual")

    def __init__(
        self,
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        vision_pre_norm=True,
        vision_post_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
        **_text_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps,
        )
        self.textual = None
        self.log_scale = mx.array(math.log(1 / 0.07))


def clip_xlm_roberta_vit_h_14(**kwargs):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
    )
    cfg.update(**kwargs)
    return XLMRobertaCLIP(**cfg)


# ---------------------------------------------------------------------------
# Preprocessing — exact equivalent of upstream CLIPModel.visual():
# bicubic resize to 224 (torch kernel, a=-0.75, align_corners=False, no
# antialias), [-1,1] -> [0,1], then CLIP mean/std normalize.
# ---------------------------------------------------------------------------

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _cubic_kernel(x, a=-0.75):
    ax = np.abs(x)
    w = np.where(
        ax <= 1,
        (a + 2) * ax**3 - (a + 3) * ax**2 + 1,
        np.where(ax < 2, a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a, 0.0),
    )
    return w


def bicubic_resize_matrix(in_size: int, out_size: int) -> mx.array:
    """[out_size, in_size] matrix M with (M @ x) == torch F.interpolate(
    x, mode='bicubic', align_corners=False) along one axis (border-clamped)."""
    scale = in_size / out_size
    M = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = (i + 0.5) * scale - 0.5
        t0 = math.floor(center)
        for tap in range(t0 - 1, t0 + 3):
            w = _cubic_kernel(center - tap)
            M[i, min(max(tap, 0), in_size - 1)] += w
    return mx.array(M.astype(np.float32))


class CLIPModel:

    def __init__(self, model=None, dtype=mx.float32):
        self.dtype = dtype
        self.model = model if model is not None else clip_xlm_roberta_vit_h_14()
        size = self.model.image_size
        self._resize_cache = {}
        self._mean = mx.array(_CLIP_MEAN).reshape(1, 3, 1, 1)
        self._std = mx.array(_CLIP_STD).reshape(1, 3, 1, 1)
        self.size = size

    def _resize(self, x):
        # x: [B, 3, H, W] -> [B, 3, size, size]
        _, _, h, w = x.shape
        key = (h, w)
        if key not in self._resize_cache:
            self._resize_cache[key] = (
                bicubic_resize_matrix(h, self.size),
                bicubic_resize_matrix(w, self.size),
            )
        mh, mw = self._resize_cache[key]
        return mx.matmul(mx.matmul(mh, x.astype(mx.float32)), mw.T)

    def visual(self, videos):
        # preprocess: list of [C, T, H, W] in [-1, 1]
        videos = mx.concatenate(
            [self._resize(u.transpose(1, 0, 2, 3)) for u in videos]
        )
        videos = (videos * 0.5 + 0.5 - self._mean) / self._std

        # forward
        out = self.model.visual(videos.astype(self.dtype), use_31_block=True)
        return out
