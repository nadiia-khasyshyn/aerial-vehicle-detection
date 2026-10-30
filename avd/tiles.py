"""Tile naming and tile layout, shared by labelling, training and eval.

A tile name carries everything needed to put its boxes back into the source
frame: `8968356_f00507_x0384_y0384`. Four scripts used to re-declare this
regex; they now all come here.
"""

import re
from typing import NamedTuple, Optional

from .config import get

TILE = int(get("tiling.tile", 512))
OVERLAP = int(get("tiling.overlap", 128))
NAME_RE = re.compile(get("tiling.name_pattern"))


class TileName(NamedTuple):
    clip: str
    frame: int
    x: int
    y: int


def parse_tile_name(stem: str) -> Optional[TileName]:
    """None when the stem is not a tile name, so callers can skip quietly."""
    match = NAME_RE.match(stem)
    if not match:
        return None
    return TileName(match.group("clip"), int(match.group("frame")),
                    int(match.group("x")), int(match.group("y")))


def tile_origins(size: int, tile: int = TILE, stride: Optional[int] = None):
    """Left/top offsets covering `size`, last tile flush with the far edge."""
    stride = stride or tile - OVERLAP
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins
