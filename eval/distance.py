"""Distance from box size, a known vehicle length and the camera FOV.

Pinhole model: an object of real length L metres spanning p pixels sits at
    d = f_px * L / p
where f_px = W / (2 * tan(hfov / 2)) is the focal length in pixels.

Assumptions, stated so they can be argued with:
  * The box's long side is the vehicle's length. True at nadir and a slight
    underestimate of p (so an overestimate of d) at oblique angles; the error
    stays under ~20% for the tilts in these clips.
  * One nominal length for every vehicle, --vehicle-len (default 4.5 m, a
    passenger car). Trucks and buses read as closer than they are; with car
    traffic dominating all five clips the band assignment is still right for
    the vast majority of boxes.
  * d is the slant range camera-to-vehicle, which is what "detection at
    300 m" means operationally.
"""

import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get

DEFAULT_HFOV_DEG = float(get("distance.hfov_deg"))
DEFAULT_VEHICLE_LEN_M = float(get("distance.vehicle_len_m"))
BAND_EDGES_M = tuple(get("distance.band_edges_m"))
BAND_NAMES = tuple(get("distance.band_names"))


def focal_px(img_w, hfov_deg=DEFAULT_HFOV_DEG):
    return img_w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def box_distance_m(x1, y1, x2, y2, img_w,
                   hfov_deg=DEFAULT_HFOV_DEG,
                   vehicle_len_m=DEFAULT_VEHICLE_LEN_M):
    long_px = max(x2 - x1, y2 - y1)
    if long_px <= 0:
        return float("inf")
    return focal_px(img_w, hfov_deg) * vehicle_len_m / long_px


def band_index(dist_m, edges=BAND_EDGES_M):
    """0-based band, or None when the distance falls outside every band."""
    for i in range(len(edges) - 1):
        if edges[i] <= dist_m < edges[i + 1]:
            return i
    return None
