"""Shared pieces of the aerial vehicle detection pipeline.

Scripts in labeling/, training/ and eval/ are entry points; everything they
have in common — the config, tile naming, box geometry — lives here so a
constant is defined once.
"""

from .config import CONFIG, get
from .tiles import TileName, parse_tile_name, tile_origins
from .boxes import (iou, iom, merge_detections, merge_overlapping, nms,
                    suppress_contained)

__all__ = ["CONFIG", "get", "TileName", "parse_tile_name", "tile_origins",
           "iou", "iom", "merge_detections", "merge_overlapping", "nms",
           "suppress_contained"]
