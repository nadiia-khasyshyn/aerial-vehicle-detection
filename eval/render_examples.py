"""Frames from a clip with the detections drawn, for looking at the output.

Yellow: ground truth, when a csv is given. Green: a detection that matches it.
Red: a detection that matches nothing, unless --all-green. Without ground truth
every detection stays green, because there is nothing to judge it against.

Frames are spaced evenly across the clip unless --frames names them.

    python eval/render_examples.py --video clip.mp4 --pred out/clip.csv \\
        --conf 0.5 --n 6 --sheet examples/clip_frames.jpg
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import iou as box_iou   # noqa: E402

GT_COLOR = (0, 215, 255)     # BGR: yellow
HIT_COLOR = (80, 230, 80)    # green
FP_COLOR = (60, 60, 255)     # red


def read_boxes(path, with_conf):
    per_frame = collections.defaultdict(list)
    if not path:
        return per_frame
    for row in csv.DictReader(open(path)):
        box = [int(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
        if with_conf:
            box.append(float(row["conf"]))
        per_frame[int(row["frame"])].append(box)
    return per_frame


def draw(frame, gt, preds, conf, iou_thr, thickness, all_green=False):
    used, hits, false_positives = set(), 0, 0
    for x1, y1, x2, y2, score in preds:
        if score < conf:
            continue
        match = next((i for i, g in enumerate(gt)
                      if i not in used and box_iou((x1, y1, x2, y2), g) >= iou_thr),
                     None)
        if gt and match is None:
            colour = HIT_COLOR if all_green else FP_COLOR
            false_positives += 1
        else:
            colour = HIT_COLOR
            used.add(match)
            hits += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, max(2, thickness // 3))
    return hits, false_positives



def crop_to(frame, boxes, pad):
    """Crop to the region the boxes occupy, padded, keeping the frame's shape.

    A 4K frame scaled into a contact-sheet panel shrinks a 60 px vehicle to a
    few pixels. Cropping to where the boxes are keeps them visible.
    """
    height, width = frame.shape[:2]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    margin_x = (x2 - x1) * pad + 40
    margin_y = (y2 - y1) * pad + 40
    x1 = max(0, int(x1 - margin_x)); x2 = min(width, int(x2 + margin_x))
    y1 = max(0, int(y1 - margin_y)); y2 = min(height, int(y2 + margin_y))
    # keep the panel aspect close to the source so the sheet stays regular
    target = width / height
    have = (x2 - x1) / max(y2 - y1, 1)
    if have < target:
        want = int((y2 - y1) * target)
        grow = (want - (x2 - x1)) // 2
        x1 = max(0, x1 - grow); x2 = min(width, x2 + grow)
    else:
        want = int((x2 - x1) / target)
        grow = (want - (y2 - y1)) // 2
        y1 = max(0, y1 - grow); y2 = min(height, y2 + grow)
    crop_to.last_offset = (x1, y1)
    return frame[y1:y2, x1:x2]


crop_to.last_offset = (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=None, help="csv: frame,x1,y1,x2,y2")
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="render exactly these frames instead of spacing them")
    ap.add_argument("--all-green", action="store_true",
                    help="draw unmatched detections green too")
    ap.add_argument("--region", type=float, nargs=4, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="crop every frame to this region, as fractions of its size")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--thickness", type=int, default=4)
    ap.add_argument("--crop-pad", type=float, default=None,
                    help="crop each frame to the boxes it holds, padded by this "
                         "fraction of the crop's size; full frames are mostly "
                         "empty road and the boxes vanish when scaled down")
    ap.add_argument("--sheet", default=None,
                    help="write one contact sheet here instead of separate files")
    ap.add_argument("--panel-width", type=int, default=760)
    ap.add_argument("--out", default="examples")
    args = ap.parse_args()

    gt = read_boxes(args.gt, with_conf=False)
    pred = read_boxes(args.pred, with_conf=True)
    frames = sorted(gt) if gt else sorted(pred)
    if not frames:
        sys.exit("no frames in the csv")
    picks = args.frames or list(dict.fromkeys(
        frames[i * (len(frames) - 1) // max(args.n - 1, 1)]
        for i in range(min(args.n, len(frames)))))

    cap = cv2.VideoCapture(args.video)
    stem = Path(args.video).stem.split("-")[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels, panel_size = [], [None]
    for index in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        boxes_here = list(gt.get(index, [])) + [
            b[:4] for b in pred.get(index, []) if b[4] >= args.conf]
        if args.region:
            h, w = frame.shape[:2]
            x1, y1 = int(args.region[0] * w), int(args.region[1] * h)
            frame = frame[y1:int(args.region[3] * h), x1:int(args.region[2] * w)]
            crop_to.last_offset = (x1, y1)
        elif args.crop_pad is not None and boxes_here:
            frame = crop_to(frame, boxes_here, args.crop_pad)
        if args.region or (args.crop_pad is not None and boxes_here):
            offset_x = crop_to.last_offset[0]
            offset_y = crop_to.last_offset[1]
            shifted_gt = [(x1 - offset_x, y1 - offset_y, x2 - offset_x, y2 - offset_y)
                          for x1, y1, x2, y2 in gt.get(index, [])]
            shifted_pred = [(x1 - offset_x, y1 - offset_y, x2 - offset_x,
                             y2 - offset_y, c) for x1, y1, x2, y2, c in pred.get(index, [])]
        else:
            shifted_gt, shifted_pred = gt.get(index, []), pred.get(index, [])
        hits, false_positives = draw(frame, shifted_gt, shifted_pred, args.conf,
                                     args.iou, args.thickness,
                                     args.all_green)
        caption = f"frame {index}  {hits + false_positives} detections"
        if gt:
            caption += (f", {false_positives} false, "
                        f"{len(gt.get(index, [])) - hits} missed")
        if panel_size[0] is None:
            panel_size[0] = (args.panel_width,
                             int(frame.shape[0] * args.panel_width / frame.shape[1]))
        frame = cv2.resize(frame, panel_size[0], interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (args.panel_width, 28), (0, 0, 0), -1)
        cv2.putText(frame, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        print(f"  {caption}")
        if args.sheet:
            panels.append(frame)
        else:
            cv2.imwrite(str(out_dir / f"{stem}_f{index:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    cap.release()

    if args.sheet and panels:
        rows = [cv2.hconcat(panels[i:i + args.cols])
                for i in range(0, len(panels), args.cols)
                if len(panels[i:i + args.cols]) == args.cols]
        sheet = Path(args.sheet)
        sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"-> {sheet}")
    elif not args.sheet:
        print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
