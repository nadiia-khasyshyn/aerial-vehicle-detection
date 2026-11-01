"""Keep only vehicle boxes that sit on road surface and that no distractor claims.

Two filters over a tiled_yoloworld detections csv:

1. **Distractor veto** — a vehicle box is dropped when a distractor detection
   (umbrella, pedestrian, manhole, road paint ...) of at least --distractor-conf
   overlaps it. The open-vocabulary head scores every prompt against the same
   region, so a confident distractor on the same pixels means the vehicle
   prompt lost.
2. **Surface test** — the pixels under and around the box must look like road,
   not vegetation. Grass is high-saturation green in HSV; asphalt is grey with
   low saturation. A box whose footprint is mostly green is dropped, which
   removes the parked-looking blobs on verges and fields.

    python filter_road_boxes.py --video clip.mp4 \
        --detections output/detections/clip.csv --out gt_boxes.csv
"""

import argparse
import collections
import csv
from pathlib import Path

import cv2
import numpy as np

VEHICLE_CLASSES = {
    "car seen from above", "van seen from above", "bus seen from above",
    "truck seen from above", "motorcycle seen from above",
    "moped seen from above", "scooter seen from above",
}


def grass_fraction(frame_hsv, box, pad_frac=0.25):
    """Share of vegetation pixels in the box and a ring around it."""
    h, w = frame_hsv.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    patch = frame_hsv[y1:y2, x1:x2]
    hue, sat, val = patch[..., 0], patch[..., 1], patch[..., 2]
    green = (hue >= 25) & (hue <= 95) & (sat >= 60) & (val >= 40)
    return float(green.mean())


def overlaps(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="minimum confidence for a vehicle box to be kept")
    ap.add_argument("--distractor-conf", type=float, default=0.2,
                    help="a distractor at or above this vetoes an overlapping vehicle")
    ap.add_argument("--distractor-overlap", type=float, default=0.3,
                    help="share of the vehicle box covered by the distractor")
    ap.add_argument("--max-grass", type=float, default=0.5,
                    help="drop a box whose footprint is greener than this")
    ap.add_argument("--max-aspect", type=float, default=2.8,
                    help="drop boxes longer than this ratio: from above they "
                         "are bike lanes, kerb walls and shadows, not vehicles")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.detections)))
    by_frame = collections.defaultdict(lambda: {"veh": [], "dis": []})
    for r in rows:
        box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        entry = (box, float(r["conf"]), r["class"])
        key = "veh" if r["class"] in VEHICLE_CLASSES else "dis"
        by_frame[int(r["frame"])][key].append(entry)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    kept, dropped_conf, dropped_dis, dropped_grass, dropped_aspect = 0, 0, 0, 0, 0

    with open(args.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "time_s", "x1", "y1", "x2", "y2", "conf", "class"])
        for frame_idx in sorted(by_frame):
            group = by_frame[frame_idx]
            if not group["veh"]:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            vetoes = [b for b, c, _ in group["dis"] if c >= args.distractor_conf]
            for box, conf, cname in group["veh"]:
                if conf < args.conf:
                    dropped_conf += 1
                    continue
                if any(overlaps(box, v) >= args.distractor_overlap for v in vetoes):
                    dropped_dis += 1
                    continue
                bw, bh = box[2] - box[0], box[3] - box[1]
                if max(bw, bh) / max(1, min(bw, bh)) > args.max_aspect:
                    dropped_aspect += 1
                    continue
                if grass_fraction(hsv, box) > args.max_grass:
                    dropped_grass += 1
                    continue
                out.writerow([frame_idx, round(frame_idx / fps, 3), *box,
                              round(conf, 4), cname])
                kept += 1

    print(f"kept {kept} vehicle boxes -> {args.out}")
    print(f"dropped: {dropped_conf} below conf, {dropped_dis} vetoed by a "
          f"distractor, {dropped_aspect} too elongated, "
          f"{dropped_grass} off the road surface")


if __name__ == "__main__":
    main()
