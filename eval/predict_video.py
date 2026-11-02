"""Run the detector over a clip, tiled, and write predictions + video.

The development path (ultralytics); `eval/detect_onnx.py` is the deployment
one and both read their thresholds from the same config.

Outputs:
  <out>/predictions.csv        frame, time_s, conf, x1, y1, x2, y2
  <out>/annotated.mp4          predictions drawn on the clip
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, merge_detections, tile_origins   # noqa: E402
from avd.video import FragmentedWriter, even_size   # noqa: E402

MERGE_FLOOR = float(get("inference.merge_floor"))
TRAIN_MEDIAN_PX = float(get("inference.train_median_px"))
SCALE_STEPS = tuple(get("inference.scale_steps"))
AUTO_TRIGGER = float(get("inference.auto_scale_trigger"))
TILE_NMS_IOU = float(get("inference.tile_nms_iou"))


def detect_frame(model, frame, tile, overlap, conf, device, batch):
    """Tiled detection at the clip's native resolution.

    Tiles are read at MERGE_FLOOR and `conf` applied only after the merge: a
    vehicle split across a seam scores higher on one fragment than on the whole
    box, so thresholding first keeps the fragment.
    """
    floor = min(conf, MERGE_FLOOR)
    h, w = frame.shape[:2]
    stride = tile - overlap
    tiles, offsets = [], []
    for y in tile_origins(h, tile, stride):
        for x in tile_origins(w, tile, stride):
            tiles.append(frame[y:y + tile, x:x + tile])
            offsets.append((x, y))

    boxes, scores = [], []
    for s in range(0, len(tiles), batch):
        results = model.predict(tiles[s:s + batch], device=device, imgsz=tile,
                                conf=floor, iou=TILE_NMS_IOU, verbose=False)
        for res, (ox, oy) in zip(results, offsets[s:s + batch]):
            b = res.boxes
            if len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            boxes.append(xyxy)
            scores.append(b.conf.cpu().numpy())

    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes, scores = merge_detections(np.concatenate(boxes),
                                     np.concatenate(scores))
    keep = scores >= conf
    return boxes[keep], scores[keep]


def choose_input_scale(model, cap, total, args, target_px):
    """Pick the input scale from the model's own detections, not from labels.

    Fires only past AUTO_TRIGGER times the trained median; reads no ground
    truth, so it needs no labelled footage to decide.
    """
    sizes = []
    for quantile in (0.2, 0.5, 0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * quantile))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes, _ = detect_frame(model, frame, args.tile, args.overlap,
                                args.conf, args.device, args.batch)
        sizes += [float(max(b[2] - b[0], b[3] - b[1])) for b in boxes]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(sizes) < 5:
        print("auto scale: too few detections to judge, staying at 1.0")
        return 1.0

    median = sorted(sizes)[len(sizes) // 2]
    trigger = target_px * AUTO_TRIGGER
    if median <= trigger:
        print(f"auto scale: median detection {median:.0f}px over {len(sizes)} "
              f"boxes, within {trigger:.0f}px — staying at 1.0")
        return 1.0
    scale = min(SCALE_STEPS, key=lambda s: abs(s - target_px / median))
    print(f"auto scale: median detection {median:.0f}px over {len(sizes)} boxes, "
          f"past {trigger:.0f}px -> scale {scale}")
    return scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="eval_run")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="chosen on the train-video val split")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this many seconds of footage (quick looks)")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="start at this timestamp in seconds")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vis-width", type=int, default=1920)
    ap.add_argument("--train-median-px", type=float, default=TRAIN_MEDIAN_PX,
                    help="median vehicle size the detector was trained on; "
                         "--input-scale auto aims for it")
    ap.add_argument("--input-scale", default="1.0",
                    help='"auto" probes a few frames and picks the scale that '
                         'brings the median detection to --train-median-px; '
                         'a number applies that scale directly. ')
    ap.add_argument("--no-video", action="store_true",
                    help="write predictions.csv only, skip the annotated mp4")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    out = Path.cwd() / args.out
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"model: {args.weights} ({n_params:,} parameters)")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vis_size = even_size(W, H, args.vis_width)
    writer = None if args.no_video else FragmentedWriter(
        out / "annotated.mp4", fps / args.frame_stride, vis_size)

    if str(args.input_scale).lower() == "auto":
        args.input_scale = choose_input_scale(model, cap, total, args,
                                              args.train_median_px)
    else:
        args.input_scale = float(args.input_scale)

    csv_f = open(out / "predictions.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "time_s", "conf", "x1", "y1", "x2", "y2"])

    first = int(args.start_s * fps)
    last = first + int(args.seconds * fps) if args.seconds else None
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    t0 = time.time()
    idx, n_proc, n_det = first - 1, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if last is not None and idx >= last:
                break
            if idx % args.frame_stride:
                continue
            if args.input_scale != 1.0:
                small = cv2.resize(frame, (int(W * args.input_scale),
                                           int(H * args.input_scale)),
                                   interpolation=cv2.INTER_AREA)
                boxes, scores = detect_frame(model, small, args.tile, args.overlap,
                                             args.conf, args.device, args.batch)
                boxes = boxes / args.input_scale
            else:
                boxes, scores = detect_frame(model, frame, args.tile, args.overlap,
                                             args.conf, args.device, args.batch)
            for (x1, y1, x2, y2), sc in zip(boxes.tolist(), scores.tolist()):
                csv_w.writerow([idx, round(idx / fps, 3), round(sc, 4),
                                int(x1), int(y1), int(x2), int(y2)])
            n_det += len(boxes)

            for (x1, y1, x2, y2), sc in zip(boxes.tolist(), scores.tolist()):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 3)
            if (frame.shape[1], frame.shape[0]) != vis_size:
                frame = cv2.resize(frame, vis_size, interpolation=cv2.INTER_AREA)
            if writer is not None:
                writer.write(frame)
            n_proc += 1
            if n_proc % 10 == 0:
                print(f"  {idx + 1}/{total} frames  "
                      f"{(time.time() - t0) / n_proc:.2f}s/frame  {n_det} dets",
                      flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_f.close()

    print(f"\n{n_proc} frames, {n_det} detections in {time.time() - t0:.0f}s")
    print(f"-> {out / 'predictions.csv'}\n-> {out / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
