# Quantization scope shared by the recipe (write side) and the pipeline
# (load side). Only the transformer-block Linears are quantized; the small
# precision-sensitive projections stay high precision (skill doctrine:
# in/out embeds, time embed, head — Lens int4 cosine 0.9944 -> 0.9976).
import json
from pathlib import Path

import mlx.nn as nn

QUANT_CONFIG_FILE = "quant.json"


def blocks_linear_predicate(path: str, module) -> bool:
    return (
        isinstance(module, nn.Linear)
        and path.startswith("blocks.")
        and module.weight.shape[-1] % 64 == 0
    )


def quantize_dit(model, bits: int, group_size: int = 64):
    nn.quantize(
        model, group_size=group_size, bits=bits, class_predicate=blocks_linear_predicate
    )
    return model


def read_quant_config(weights_dir) -> dict | None:
    p = Path(weights_dir) / QUANT_CONFIG_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_quant_config(weights_dir, bits: int, group_size: int, dit_file: str):
    p = Path(weights_dir) / QUANT_CONFIG_FILE
    p.write_text(
        json.dumps(
            {
                "bits": bits,
                "group_size": group_size,
                "dit_file": dit_file,
                "predicate": "blocks_linears",
            },
            indent=2,
        )
    )
