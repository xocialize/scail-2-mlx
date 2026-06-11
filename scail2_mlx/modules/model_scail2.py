# MLX port of refs/SCAIL-2/wan/modules/model_scail2.py (zai-org/SCAIL-2,
# branch wan-scail2). Isomorphic translation: same module/function names, same
# decomposition, same forward order. Only PT<->MLX substitutions:
#   - complex RoPE (torch.polar, float64) -> (cos, sin) pairs in float32
#   - Conv3d NCDHW -> NDHWC at the patch-embedding call sites
#   - flash_attention -> mx.fast.scaled_dot_product_attention shim
#   - nn.Sequential indices "0/2" -> mlx Sequential "layers.0/layers.2"
#     (weight loaders remap keys; see utils/weights.py)
import math
from functools import reduce
from operator import mul

import mlx.core as mx
import mlx.nn as nn

from .attention import flash_attention

__all__ = ["SCAIL2Model"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)

    # calculation
    sinusoid = position[:, None] * mx.power(
        10000, -(mx.arange(half).astype(mx.float32) / half)
    )[None, :]
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    # Upstream returns torch.polar(1, freqs) (complex). Here: (cos, sin) pair.
    assert dim % 2 == 0
    freqs = mx.arange(max_seq_len).astype(mx.float32)[:, None] * (
        1.0 / mx.power(theta, mx.arange(0, dim, 2).astype(mx.float32) / dim)
    )[None, :]
    return mx.cos(freqs), mx.sin(freqs)


def _split_freqs(freqs_cos, freqs_sin, c):
    # upstream: freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    cos_parts = (
        freqs_cos[:, :s0],
        freqs_cos[:, s0 : s0 + s1],
        freqs_cos[:, s0 + s1 :],
    )
    sin_parts = (
        freqs_sin[:, :s0],
        freqs_sin[:, s0 : s0 + s1],
        freqs_sin[:, s0 + s1 :],
    )
    return cos_parts, sin_parts, (s0, s1, s1)


def _freqs_grid(parts, sizes, f, h, w, shift_f, shift_h, shift_w):
    # upstream freqs_i: cat of t/h/w slices expanded to (f, h, w, -1)
    p_t, p_h, p_w = parts
    s0, s1, s2 = sizes
    return mx.concatenate(
        [
            mx.broadcast_to(p_t[shift_f : shift_f + f].reshape(f, 1, 1, s0), (f, h, w, s0)),
            mx.broadcast_to(p_h[shift_h : shift_h + h].reshape(1, h, 1, s1), (f, h, w, s1)),
            mx.broadcast_to(p_w[shift_w : shift_w + w].reshape(1, 1, w, s2), (f, h, w, s2)),
        ],
        axis=-1,
    )


def _rotate(x, cos_grid, sin_grid):
    # x: [B, S, N, D] -> complex pairs on adjacent lanes (view_as_complex semantics)
    b, s, n, d = x.shape
    x = x.astype(mx.float32).reshape(b, s, n, d // 2, 2)
    xr = x[..., 0]
    xi = x[..., 1]
    # grids: [S, c] -> broadcast over batch and heads
    fc = cos_grid.reshape(1, s, 1, -1)
    fs = sin_grid.reshape(1, s, 1, -1)
    out_r = xr * fc - xi * fs
    out_i = xr * fs + xi * fc
    return mx.stack([out_r, out_i], axis=-1).reshape(b, s, n, d)


def rope_apply_ref(x, freqs, **kwargs):
    f = 1
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = kwargs["rope_T_shift"]["ref"]
    shift_h = kwargs["rope_H_shift"]["ref"]
    shift_w = kwargs["rope_W_shift"]["ref"]

    c = x.shape[3] // 2
    cos_parts, sin_parts, sizes = _split_freqs(freqs[0], freqs[1], c)

    seq_len = f * h * w
    assert seq_len == x.shape[1]

    cos_grid = _freqs_grid(cos_parts, sizes, f, h, w, shift_f, shift_h, shift_w).reshape(seq_len, -1)
    sin_grid = _freqs_grid(sin_parts, sizes, f, h, w, shift_f, shift_h, shift_w).reshape(seq_len, -1)
    return _rotate(x, cos_grid, sin_grid)


def rope_apply_video(x, freqs, **kwargs):
    f = kwargs["rope_T"]
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = kwargs["rope_T_shift"]["video"]
    shift_h = kwargs["rope_H_shift"]["video"]
    shift_w = kwargs["rope_W_shift"]["video"]

    c = x.shape[3] // 2
    cos_parts, sin_parts, sizes = _split_freqs(freqs[0], freqs[1], c)

    seq_len = f * h * w
    assert seq_len == x.shape[1]

    cos_grid = _freqs_grid(cos_parts, sizes, f, h, w, shift_f, shift_h, shift_w).reshape(seq_len, -1)
    sin_grid = _freqs_grid(sin_parts, sizes, f, h, w, shift_f, shift_h, shift_w).reshape(seq_len, -1)
    return _rotate(x, cos_grid, sin_grid)


def rope_apply_pose(x, freqs, **kwargs):
    f = kwargs["rope_T"]
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = kwargs["rope_T_shift"]["pose"]
    shift_h = kwargs["rope_H_shift"]["pose"]
    shift_w = kwargs["rope_W_shift"]["pose"]

    c = x.shape[3] // 2
    cos_parts, sin_parts, sizes = _split_freqs(freqs[0], freqs[1], c)

    seq_len = f * (h // 2) * (w // 2)  # downsampled
    assert seq_len == x.shape[1]
    # upstream F.avg_pool2d(k=2, s=2) floors odd trailing rows/cols; the
    # divisible-by-32 input constraint keeps the rope grid even.
    assert h % 2 == 0 and w % 2 == 0

    cos_grid = _freqs_grid(cos_parts, sizes, f, h, w, shift_f, shift_h, shift_w)
    sin_grid = _freqs_grid(sin_parts, sizes, f, h, w, shift_f, shift_h, shift_w)

    # downsample: avg_pool2d(k=2, s=2) over (h, w), on real and imag separately
    c_total = cos_grid.shape[-1]
    cos_grid = cos_grid.reshape(f, h // 2, 2, w // 2, 2, c_total).mean(axis=(2, 4))
    sin_grid = sin_grid.reshape(f, h // 2, 2, w // 2, 2, c_total).mean(axis=(2, 4))

    cos_grid = cos_grid.reshape(seq_len, -1)
    sin_grid = sin_grid.reshape(seq_len, -1)
    return _rotate(x, cos_grid, sin_grid)


def rope_apply_scail(x, **kwargs):
    """
    x: [b, s, n, d]
    """
    ref_length = kwargs["ref_length"]
    video_length = kwargs["seq_length"]
    pose_length = kwargs["pose_length"]

    x_ref = x[:, :ref_length]
    x_video = x[:, ref_length : ref_length + video_length]
    x_pose = x[:, -pose_length:]

    return mx.concatenate(
        [
            rope_apply_ref(x_ref, **kwargs),
            rope_apply_video(x_video, **kwargs),
            rope_apply_pose(x_pose, **kwargs),
        ],
        axis=1,
    )


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.astype(mx.float32)).astype(x.dtype) * self.weight

    def _norm(self, x):
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)


class WanLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        weight = getattr(self, "weight", None)
        bias = getattr(self, "bias", None)
        return mx.fast.layer_norm(x.astype(mx.float32), weight, bias, self.eps).astype(
            x.dtype
        )


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(self, x, seq_lens, rope_apply_func, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # compute in the weight dtype (upstream amp.autocast(bf16) equivalent;
        # no-op when weights are fp32, e.g. in parity tests)
        x = x.astype(self.q.weight.dtype)

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).reshape(b, s, n, d)
            k = self.norm_k(self.k(x)).reshape(b, s, n, d)
            v = self.v(x).reshape(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply_func(q),
            k=rope_apply_func(k),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.reshape(b, s, -1)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def __call__(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        x = x.astype(self.q.weight.dtype)
        context = context.astype(self.q.weight.dtype)

        # compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.reshape(b, x.shape[1], -1)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        x = x.astype(self.q.weight.dtype)
        context = context.astype(self.q.weight.dtype)
        context_img = context_img.astype(self.q.weight.dtype)

        # compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).reshape(b, -1, n, d)
        v_img = self.v_img(context_img).reshape(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.reshape(b, x.shape[1], -1)
        img_x = img_x.reshape(b, img_x.shape[1], -1)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approx="tanh"), nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = mx.random.normal((1, 6, dim)) / dim**0.5

    def __call__(
        self,
        x,
        e,
        seq_lens,
        context,
        context_lens,
        **kwargs,  # contains rope_apply_func
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
        """
        # modulation kept float32 (upstream amp.autocast(float32) region)
        e = mx.split(
            (self.modulation.astype(mx.float32) + e.astype(mx.float32)), 6, axis=1
        )

        # self-attention
        y = self.self_attn(
            self.norm1(x).astype(mx.float32) * (1 + e[1]) + e[0],
            seq_lens,
            **kwargs,
        )
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            ffn_dtype = self.ffn.layers[0].weight.dtype
            y = self.ffn(
                (self.norm2(x).astype(mx.float32) * (1 + e[4]) + e[3]).astype(ffn_dtype)
            )
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = mx.random.normal((1, 2, dim)) / dim**0.5

    def __call__(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = mx.split(
            self.modulation.astype(mx.float32) + e.astype(mx.float32)[:, None, :],
            2,
            axis=1,
        )
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = mx.zeros((1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def __call__(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.reshape(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


def _conv3d_ncdhw(conv, x):
    # PT Conv3d takes NCDHW; MLX takes NDHWC. Returns NDHWC output.
    # Input cast to weight dtype (autocast equivalent; no-op at fp32).
    return conv(x.transpose(0, 2, 3, 4, 1).astype(conv.weight.dtype))


class SCAIL2Model(nn.Module):
    r"""
    SCAIL2 diffusion backbone.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        mask_dim=28,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        pose_rope_shift=[0, 0, 120],  # shift in (t, h, w) for pose rope embedding
        eps=1e-6,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v", "vace"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.mask_dim = mask_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.pose_rope_shift = pose_rope_shift
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.patch_embedding_pose = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.patch_embedding_mask = nn.Conv3d(
            mask_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approx="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = [
            WanAttentionBlock(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
            )
            for _ in range(num_layers)
        ]

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # rope freqs as (cos, sin); upstream keeps complex freqs out of
        # state_dict ("don't use register_buffer"), so these are plain
        # attributes excluded from parameters via leading underscore
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        f_t = rope_params(8192, d - 4 * (d // 6))
        f_h = rope_params(8192, 2 * (d // 6))
        f_w = rope_params(8192, 2 * (d // 6))
        self._freqs_cos = mx.concatenate([f_t[0], f_h[0], f_w[0]], axis=1)
        self._freqs_sin = mx.concatenate([f_t[1], f_h[1], f_w[1]], axis=1)
        self.hidden_size_head = d

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

    @property
    def freqs(self):
        return (self._freqs_cos, self._freqs_sin)

    def apply_i2v_ones_masks(self, inputs, mask_dim: int = 4):
        b, d, t, h, w = inputs.shape
        mask = mx.ones((b, mask_dim, t, h, w), dtype=inputs.dtype)
        inputs = mx.concatenate([inputs, mask], axis=1)
        return inputs

    def apply_i2v_zeros_masks(self, inputs, mask_dim: int = 4):
        b, d, t, h, w = inputs.shape
        mask = mx.zeros((b, mask_dim, t, h, w), dtype=inputs.dtype)
        inputs = mx.concatenate([inputs, mask], axis=1)
        return inputs

    def merge_list_of_tensors_to_batch(self, inputs):
        return mx.concatenate([mx.expand_dims(u, 0) for u in inputs], axis=0)

    def __call__(
        self,
        x,
        pose_latents,
        driving_masks,
        ref_latents,
        ref_masks,
        t,
        context,
        seq_len,
        replace_flag: bool,
        history_mask=None,
        clip_fea=None,
    ):
        r"""
        Forward pass through the diffusion model. Argument semantics identical
        to upstream SCAIL2Model.forward (lists of [C, F, H, W] tensors).
        """
        assert clip_fea is not None

        x = self.merge_list_of_tensors_to_batch(x)
        ref_latents = self.merge_list_of_tensors_to_batch(ref_latents)
        pose_latents = self.merge_list_of_tensors_to_batch(pose_latents)
        driving_masks = self.merge_list_of_tensors_to_batch(driving_masks)
        ref_masks = self.merge_list_of_tensors_to_batch(ref_masks)

        if history_mask is None:
            x = self.apply_i2v_zeros_masks(x)
        else:
            history_mask = self.merge_list_of_tensors_to_batch(history_mask)
            x = mx.concatenate([x, history_mask], axis=1)
        ref_latents = self.apply_i2v_ones_masks(ref_latents)
        pose_latents = self.apply_i2v_ones_masks(pose_latents)

        B, D, T, H, W = x.shape

        assert pose_latents.shape[3] == H // 2
        assert pose_latents.shape[4] == W // 2

        ref_length = 1 * H * W // reduce(mul, self.patch_size)
        seq_length = T * ref_length
        pose_length = T * (H // 2) * (W // 2) // reduce(mul, self.patch_size)

        # embeddings (convs run NDHWC; outputs stay NDHWC = "b t h w c")
        x = mx.concatenate([ref_latents, x], axis=2)
        x = _conv3d_ncdhw(self.patch_embedding, x)
        ref_mask_emb = _conv3d_ncdhw(self.patch_embedding_mask, ref_masks)
        x = x + ref_mask_emb
        pose_emb = _conv3d_ncdhw(self.patch_embedding_pose, pose_latents)
        sam_emb = _conv3d_ncdhw(self.patch_embedding_mask, driving_masks)
        pose_emb = pose_emb + sam_emb
        # "b c t h w -> b (t h w) c" on NDHWC is a plain reshape
        x = mx.concatenate(
            [
                x.reshape(B, -1, self.dim),
                pose_emb.reshape(B, -1, self.dim),
            ],
            axis=1,
        )

        seq_lens = [x.shape[1]] * B

        # time embeddings (float32)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).astype(mx.float32)
        )
        e0 = self.time_projection(e).reshape(-1, 6, self.dim)

        # context
        context_lens = None
        context = self.text_embedding(
            mx.stack(
                [
                    mx.concatenate(
                        [u, mx.zeros((self.text_len - u.shape[0], u.shape[1]), dtype=u.dtype)]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = mx.concatenate([context_clip, context], axis=1)

        rope_t = T // self.patch_size[0]
        rope_h = H // self.patch_size[1]
        rope_w = W // self.patch_size[2]

        grid_sizes = [(rope_t, rope_h, rope_w) for _ in range(B)]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            context=context,
            context_lens=context_lens,
            freqs=self.freqs,
            ref_length=ref_length,
            seq_length=seq_length,
            pose_length=pose_length,
        )

        kwargs["rope_T"] = rope_t
        kwargs["rope_H"] = rope_h
        kwargs["rope_W"] = rope_w
        kwargs["hidden_size_head"] = self.hidden_size_head

        kwargs["rope_T_shift"] = {
            "ref": 0,
            "pose": 0 if replace_flag else 1,
            "video": 0 if replace_flag else 1,
        }

        kwargs["rope_H_shift"] = {
            "ref": 120 if replace_flag else 0,
            "pose": 0,
            "video": 0,
        }

        kwargs["rope_W_shift"] = {
            "ref": 0,
            "pose": 120,
            "video": 0,
        }

        def apply_rope_scail(x):
            """
            x: [b, s, n, d]
            """
            return rope_apply_scail(x, **kwargs)

        kwargs["rope_apply_func"] = apply_rope_scail

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, offset=ref_length)
        return [u.astype(mx.float32) for u in x]

    def unpatchify(self, x, grid_sizes, offset: int = 0):
        r"""
        Reconstruct video tensors from patch embeddings. Keeps only the
        denoised video segment (skips ref tokens via offset; pose tokens fall
        beyond offset + prod(grid) and are dropped).
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes):
            u = u[offset : offset + math.prod(v)].reshape(*v, *self.patch_size, c)
            # einsum 'fhwpqrc->cfphqwr'
            u = u.transpose(6, 0, 3, 1, 4, 2, 5)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
