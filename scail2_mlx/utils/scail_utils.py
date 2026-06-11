# MLX port of refs/SCAIL-2/wan/utils/scail_utils.py (tensor math only; the
# decord/PIL loading helpers belong to the pipeline layer).
import mlx.core as mx

__all__ = ["extract_and_compress_mask_to_latent"]


def extract_and_compress_mask_to_latent(
    mask_cthw, additional_spatial_downsample=1, temporal_compression_stride=4
):
    """RGB segmentation mask (3, T, H, W) in [-1, 1] -> 28-channel binary
    latent (28, T_latent, H_latent, W_latent) in {0, 1}, no VAE involved.

    7 color-coded channels (white/red/green/blue/yellow/magenta/cyan) x
    temporal_compression_stride frames packed per latent frame.
    """
    C, T, H, W = mask_cthw.shape
    _ON_THRESH = (225.0 - 127.5) / 127.5  # pixel >= 225 counts as "on"
    mask = mask_cthw.transpose(1, 0, 2, 3).astype(mx.float32)  # (T, 3, H, W)
    R = (mask[:, 0:1] > _ON_THRESH).astype(mx.float32)
    G = (mask[:, 1:2] > _ON_THRESH).astype(mx.float32)
    B = (mask[:, 2:3] > _ON_THRESH).astype(mx.float32)
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = mx.concatenate(
        [
            R * G * B,
            R * nG * nB,
            nR * G * nB,
            nR * nG * B,
            R * G * nB,
            R * nG * B,
            nR * G * B,
        ],
        axis=1,
    )  # (T, 7, H, W)

    H_lat, W_lat = H, W
    if additional_spatial_downsample > 1:
        H_lat = H_lat // additional_spatial_downsample
        W_lat = W_lat // additional_spatial_downsample
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2

    # upstream F.interpolate(mode='area') == uniform mean pooling when the
    # ratio is integral; the divisible-by-32 input constraint guarantees that
    assert H % H_lat == 0 and W % W_lat == 0, (
        f"area interpolate needs integral ratio, got {H}x{W} -> {H_lat}x{W_lat}"
    )
    kh, kw = H // H_lat, W // W_lat
    t, c7 = binary_7ch.shape[0], binary_7ch.shape[1]
    binary_7ch = binary_7ch.reshape(t, c7, H_lat, kh, W_lat, kw).mean(axis=(3, 5))

    T_latent = (T - 1) // temporal_compression_stride + 1
    padded = mx.concatenate(
        [
            mx.repeat(binary_7ch[:1], temporal_compression_stride, axis=0),
            binary_7ch[1:],
        ],
        axis=0,
    )
    out = padded.reshape(
        T_latent, temporal_compression_stride * 7, H_lat, W_lat
    ).transpose(1, 0, 2, 3)
    return out  # (28, T_latent, H_lat, W_lat)
