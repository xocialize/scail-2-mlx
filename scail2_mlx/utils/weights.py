# Weight-key mapping between upstream SCAIL-2 PT state_dict names (already in
# Wan layout, i.e. the output of refs/SCAIL-2/convert.py) and this port's MLX
# module tree. Differences are purely structural:
#   - mlx nn.Sequential children live under ".layers."
#   - Conv3d weights transpose (O, I, kT, kH, kW) -> (O, kT, kH, kW, I)
import re

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

_SEQUENTIAL_PREFIXES = (
    "text_embedding",
    "time_embedding",
    "time_projection",
    "ffn",
    r"img_emb\.proj",
)
_SEQ_RE = re.compile(
    r"\b(" + "|".join(_SEQUENTIAL_PREFIXES) + r")\.(\d+)\."
)

_CONV3D_RE = re.compile(
    r"^(patch_embedding|patch_embedding_pose|patch_embedding_mask)\.weight$"
)


def pt_key_to_mlx(key: str) -> str:
    return _SEQ_RE.sub(lambda m: f"{m.group(1)}.layers.{m.group(2)}.", key)


def is_conv3d_weight(key: str) -> bool:
    return _CONV3D_RE.match(key) is not None


def load_state_dict_np(model, state: dict, strict: bool = True) -> None:
    """Load {pt_key: np.ndarray} into an MLX SCAIL2Model."""
    flat = []
    for key, arr in state.items():
        arr = np.ascontiguousarray(arr)
        if is_conv3d_weight(key):
            arr = arr.transpose(0, 2, 3, 4, 1)
        flat.append((pt_key_to_mlx(key), mx.array(arr)))

    if strict:
        expected = {k for k, _ in tree_flatten(model.parameters())}
        got = {k for k, _ in flat}
        missing = expected - got
        unexpected = got - expected
        if missing or unexpected:
            raise ValueError(
                f"state_dict mismatch: missing={sorted(missing)[:8]} "
                f"unexpected={sorted(unexpected)[:8]}"
            )

    model.update(tree_unflatten(flat))
    mx.eval(model.parameters())
