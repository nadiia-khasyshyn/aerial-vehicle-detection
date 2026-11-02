"""One frame at several confidence thresholds, side by side.

Yellow: ground truth. Green: a detection that matches one. Red: a detection
that matches nothing — a false positive, if the ground truth is complete.

Shows what the shipping threshold buys and costs: the same detections, cut at
each threshold, so the reader can see which boxes go first. The frame is
detected once at the lowest threshold and then filtered — the pipeline applies
the threshold after the seam merge, so this is identical to running it three
times and far cheaper.

    python eval/render_thresholds.py --video clip.mp4 --frame 0.5 \
        --gt gt.csv --out examples/val_thresholds.jpg
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, iou as box_iou   # noqa: E402
from avd.onnx_detector import OnnxDetector   # noqa: E402
from eval.detect_onnx import detect_frame   # noqa: E402

TILE = int(get("tiling.tile"))
OVERLAP = int(get("tiling.overlap"))
GT_COLOR = (0, 215, 255)     # BGR: yellow, ground truth
DET_COLOR = (80, 230, 80)    # green, matches a ground-truth box
FP_COLOR = (60, 60, 255)     # red, matches none


def read_gt(path):
    per_frame = collections.defaultdict(list)
    if not path:
        return per_frame
    for row in csv.DictReader(open(path)):
        per_frame[int(row["frame"])].append(
            tuple(int(float(row[k])) for k in ("x1", "y1", "x2", "y2")))
    return per_frame


def panel(frame, boxes, gt, conf, width, thickness, iou_thr):
    """Green where a detection matches ground truth, red where it matches none.

    Without ground truth there is nothing to judge a box against, so every
    detection stays green and the caption says so.
    """
    img = frame.copy()
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, thickness)

    used, false_positives = set(), 0
    for box in boxes:
        hit = next((i for i, g in enumerate(gt)
                    if i not in used and box_iou(box, g) >= iou_thr), None)
        if gt and hit is None:
            colour = FP_COLOR
            false_positives += 1
        else:
            colour = DET_COLOR
            used.add(hit)
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      colour, thickness)

    height = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    caption = f"conf {conf:.2f} - {len(boxes)} detections"
    if gt:
        caption += f", {false_positives} false"
    cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(img, caption, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="weights/vehicle_detector.onnx")
    ap.add_argument("--frame", default="0.5",
                    help="frame index, or a fraction of the clip's length")
    ap.add_argument("--gt", default=None, help="csv: frame,x1,y1,x2,y2")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.35, 0.50, 0.75])
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU at which a detection counts as matching")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--panel-width", type=int, default=760)
    ap.add_argument("--thickness", type=int, default=5)
    ap.add_argument("--stack", choices=["h", "v"], default="h")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = (int(float(args.frame) * total) if "." in str(args.frame)
             else int(args.frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read frame {index} of {args.video}")

    detector = OnnxDetector(args.model)
    floor = min(args.conf)
    if args.input_scale != 1.0:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * args.input_scale),
                                   int(h * args.input_scale)),
                           interpolation=cv2.INTER_AREA)
        boxes, scores = detect_frame(detector, small, args.tile, args.overlap,
                                     floor, args.batch)
        boxes = boxes / args.input_scale
    else:
        boxes, scores = detect_frame(detector, frame, args.tile, args.overlap,
                                     floor, args.batch)

    gt = read_gt(args.gt)[index]
    panels = [panel(frame, boxes[scores >= c].tolist(), gt, c,
                    args.panel_width, args.thickness, args.iou)
              for c in sorted(args.conf)]

    sheet = cv2.hconcat(panels) if args.stack == "h" else cv2.vconcat(panels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    counts = ", ".join(f"{c:.2f}: {int((scores >= c).sum())}"
                       for c in sorted(args.conf))
    print(f"{out}  frame {index} of {total}, {len(gt)} ground-truth boxes")
    print(f"  detections by threshold -> {counts}")


if __name__ == "__main__":
    main()
