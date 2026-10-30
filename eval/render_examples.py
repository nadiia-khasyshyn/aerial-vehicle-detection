"""Save a few eval frames with ground truth and predictions drawn together.

Green: ground truth. Red: predictions at the shipping threshold. A correct
detection shows as red-on-green; a miss as bare green; a false alarm as bare
red. Frames are picked evenly across the clip so the set is not cherry-picked.
"""

import argparse
import collections
import csv
from pathlib import Path

import cv2


def read_boxes(path, with_conf):
    per_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        box = [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]
        if with_conf:
            box.append(float(r["conf"]))
        per_frame[int(r["frame"])].append(box)
    return per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="examples")
    args = ap.parse_args()

    gt = read_boxes(args.gt, with_conf=False)
    pred = read_boxes(args.pred, with_conf=True)
    frames = sorted(set(gt))
    picks = [frames[i * (len(frames) - 1) // max(args.n - 1, 1)]
             for i in range(min(args.n, len(frames)))]
    out = Path.cwd() / args.out
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    stem = Path(args.video).stem.split("-")[0]
    for f in dict.fromkeys(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        for x1, y1, x2, y2 in gt.get(f, []):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        n_pred = 0
        for x1, y1, x2, y2, c in pred.get(f, []):
            if c < args.conf:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            n_pred += 1
        label = f"frame {f}   GT {len(gt.get(f, []))} (green)   pred {n_pred} (red)"
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_f{f:05d}_gt_vs_pred.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  frame {f}: {len(gt.get(f, []))} GT, {n_pred} predictions")
    cap.release()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
