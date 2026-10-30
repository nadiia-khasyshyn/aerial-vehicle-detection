"""Render the frames the dataset was cut from, showing each tile and the
detections that survived classification inside it.

Boxes are drawn only inside their own tile: the rest of the frame was never
looked at for that tile, so drawing anything there would imply a judgement
that was never made. Which crops count as valid is read from the accepted
folder on disk, so anything moved out by hand is excluded automatically.
"""

import argparse
import collections
import csv
import re
from pathlib import Path

import cv2

from tiled_yoloworld import FragmentedWriter

VIDEOS = {
    "3405804": "3405804-uhd_3840_2160_30fps.mp4",
    "5382494": "5382494-uhd_3840_2160_24fps.mp4",
    "8457857": "8457857-uhd_3840_2160_24fps.mp4",
    "8968356": "8968356-hd_1920_1080_30fps.mp4",
}
# _b<i>_p... classifier crops, _w<i>_<cls>_c... YOLO-World additions,
# _c<i>_c.../_y<i>_c... the single-pass pipeline (custom / YOLOE),
# _g<i>_<src>_c... boxes the old area cap had deleted,
# _m<i>_h_c... boxes drawn by hand over the tiles.
NAME_RE = re.compile(r"^(?P<vid>\d+)_f(?P<frame>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_"
                     r"(?:b\d+_p|w\d+_\w+_c|g\d+_\w+_c|m\d+_\w+_c|[cy]\d+_c)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", nargs="+",
                    default=["crops_classified/vehicle_ge080",
                             "crops_classified/added_yw"],
                    help="every folder whose crops count as valid")
    ap.add_argument("--scores", nargs="+",
                    default=["crops_classified/scores.csv"],
                    help="every csv holding box coordinates for those crops")
    ap.add_argument("--out", default="output_valid")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=1920)
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out / "videos"
    out.mkdir(parents=True, exist_ok=True)

    kept = {p.name for d in args.accepted
            for p in (root / d).glob("*.jpg")}
    # frame -> tile origin -> boxes, per video
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    n_box = 0
    for r in (r for f in args.scores
              for r in csv.DictReader(open(root / f))):
        if r["crop"] not in kept:
            continue
        m = NAME_RE.match(r["crop"])
        if not m:
            continue
        vid, f = m["vid"], int(m["frame"])
        ox, oy = int(m["x"]), int(m["y"])
        box = (ox + int(r["x1"]), oy + int(r["y1"]),
               ox + int(r["x2"]), oy + int(r["y2"]))
        data[vid][(f, ox, oy)].append(box)
        n_box += 1
    print(f"{len(kept)} accepted crops, {n_box} boxes placed")

    for vid, per_tile in sorted(data.items()):
        by_frame = collections.defaultdict(list)
        for (f, ox, oy), boxes in per_tile.items():
            by_frame[f].append((ox, oy, boxes))

        cap = cv2.VideoCapture(str(root / VIDEOS[vid]))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = args.width / W if W > args.width else 1.0
        vw, vh = int(W * scale), int(H * scale)
        writer = FragmentedWriter(out / f"{vid}_valid.mp4", args.fps, (vw, vh))

        wanted = sorted(by_frame)
        idx = -1
        written = 0
        try:
            for f in wanted:
                while idx < f:
                    if not cap.grab():
                        break
                    idx += 1
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                for ox, oy, boxes in by_frame[f]:
                    cv2.rectangle(frame, (ox, oy), (ox + args.tile, oy + args.tile),
                                  (0, 215, 255), 3)
                    for x1, y1, x2, y2 in boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{vid}  frame {f}  tiles {len(by_frame[f])}  " \
                        f"boxes {sum(len(b) for _, _, b in by_frame[f])}"
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (255, 255, 255), 2, cv2.LINE_AA)
                if scale != 1.0:
                    frame = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
        finally:
            cap.release()
            writer.release()
        print(f"  {vid}: {written} frames, {len(per_tile)} tiles, "
              f"{sum(len(b) for b in per_tile.values())} boxes")

    print(f"videos: {out}")


if __name__ == "__main__":
    main()
