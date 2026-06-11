# Chunked (per-latent-frame) Wan2.1 VAE decode with causal feature caching,
# mirroring upstream refs/SCAIL-2/wan/modules/vae.py decode().
#
# mlx-video's Decoder3d only implements whole-sequence decode, whose
# upsample3d path temporally doubles EVERY frame (4*T outputs) — upstream's
# cached path marks the first chunk 'Rep' and skips time_conv, yielding the
# canonical 1+(T-1)*4 frames and different boundary frames. All cached
# building blocks (CausalConv3d cache_x, ResidualBlock feat_cache) already
# exist in mlx-video; this module only adds the missing orchestration +
# the upsample3d Rep-sentinel branch. No upstream edits (free-function
# bypass, lance vae_stream.py precedent).
import mlx.core as mx

from mlx_video.models.wan_2.vae import (
    CACHE_T,
    Resample,
    ResidualBlock,
)

__all__ = ["decode_chunked", "count_decoder_cache_slots"]

_REP = "Rep"


def count_decoder_cache_slots(decoder) -> int:
    n = 1  # decoder.conv1
    for layer in decoder.middle:
        if isinstance(layer, ResidualBlock):
            n += 2
    for layer in decoder.upsamples:
        if isinstance(layer, ResidualBlock):
            n += 2
        elif isinstance(layer, Resample) and layer.mode == "upsample3d":
            n += 1
    n += 1  # decoder.head[2]
    return n


def _conv_cached(conv, x, feat_cache, feat_idx):
    # CausalConv3d with rolling 2-frame cache (encoder conv1/head pattern)
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:]
    if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
        cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
    out = conv(x, cache_x=feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return out


def _resample_cached(layer, x, feat_cache, feat_idx):
    # upstream Resample.forward (upsample modes) with the 'Rep' first-chunk
    # sentinel: chunk 0 skips time_conv (no temporal doubling of frame 0),
    # chunk 1 runs it zero-padded, chunk >=2 runs it against the cache
    b, c, t, h, w = x.shape

    if layer.mode == "upsample3d":
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = _REP
            feat_idx[0] += 1
        else:
            cache_x = x[:, :, -CACHE_T:]
            was_rep = isinstance(feat_cache[idx], str)
            if cache_x.shape[2] < 2 and not was_rep:
                cache_x = mx.concatenate(
                    [feat_cache[idx][:, :, -1:], cache_x], axis=2
                )
            elif cache_x.shape[2] < 2 and was_rep:
                cache_x = mx.concatenate([mx.zeros_like(cache_x), cache_x], axis=2)
            if was_rep:
                x_t = layer.time_conv(x)
            else:
                x_t = layer.time_conv(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            x_t = x_t.reshape(b, 2, c, t, h, w)
            x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
            t = t * 2

    # spatial 2x: nearest + Conv2d (identical to mlx-video Resample)
    x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
    x = mx.repeat(x, 2, axis=1)
    x = mx.repeat(x, 2, axis=2)
    x = layer.resample[1](x)
    c_out = x.shape[-1]
    return x.reshape(b, t, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)


def _decoder_chunk(decoder, x, feat_cache, feat_idx):
    import mlx.nn as nn

    x = _conv_cached(decoder.conv1, x, feat_cache, feat_idx)

    for layer in decoder.middle:
        if isinstance(layer, ResidualBlock):
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = layer(x)

    for layer in decoder.upsamples:
        if isinstance(layer, ResidualBlock):
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = _resample_cached(layer, x, feat_cache, feat_idx)

    x = nn.silu(decoder.head[0](x))
    x = _conv_cached(decoder.head[2], x, feat_cache, feat_idx)
    return x


def decode_chunked(model, z: mx.array) -> mx.array:
    """Upstream-faithful decode: z [B, 16, T, h, w] (normalized) ->
    video [B, 3, 1+(T-1)*4, H, W] in [-1, 1]. Flat memory in T."""
    mean = model.mean.reshape(1, -1, 1, 1, 1)
    inv_std = model.inv_std.reshape(1, -1, 1, 1, 1)
    z = z / inv_std + mean

    x = model.conv2(z)  # 1x1x1, no temporal context

    feat_cache = [None] * count_decoder_cache_slots(model.decoder)
    out = []
    for i in range(x.shape[2]):
        feat_idx = [0]
        chunk = _decoder_chunk(model.decoder, x[:, :, i : i + 1], feat_cache, feat_idx)
        mx.eval(chunk, *[c for c in feat_cache if isinstance(c, mx.array)])
        out.append(chunk)
    return mx.clip(mx.concatenate(out, axis=2), -1, 1)
