"""Tiled detection over a folder of videos, on onnxruntime alone.

The deployment entry point: no torch, no ultralytics. Tiling, the seam merge
and the thresholds all come from `configs/pipeline.yaml`.

    python eval/detect_onnx.py --in videos/in --out videos/out

Each clip produces <name>.mp4 with boxes drawn and <name>.csv with
frame, time_s, conf, x1, y1, x2, y2.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, merge_detections, tile_origins   # noqa: E402
from avd.onnx_detector import OnnxDetector   # noqa: E402
from avd.video import FragmentedWriter, even_size   # noqa: E402

MERGE_FLOOR = float(get("inference.merge_floor"))
TRAIN_MEDIAN_PX = float(get("inference.train_median_px"))
SCALE_STEPS = tuple(get("inference.scale_steps"))
AUTO_TRIGGER = float(get("inference.auto_scale_trigger"))
TILE = int(get("tiling.tile"))
OVERLAP = int(get("tiling.overlap"))
SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def detect_frame(detector, frame, tile, overlap, conf, batch):
    """Tiled detection in frame coordinates, thresholded after the seam merge."""
    floor = min(conf, MERGE_FLOOR)
    h, w = frame.shape[:2]
    stride = tile - overlap
    crops, offsets = [], []
    for y in tile_origins(h, tile, stride):
        for x in tile_origins(w, tile, stride):
            crops.append(frame[y:y + tile, x:x + tile])
            offsets.append((x, y))

    boxes, scores = [], []
    for start in range(0, len(crops), batch):
        chunk = crops[start:start + batch]
        for (b, s), (ox, oy) in zip(detector.detect_tiles(chunk, floor),
                                    offsets[start:start + batch]):
            if not len(b):
                continue
            b = b.copy()
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            boxes.append(b)
            scores.append(s)

    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes, scores = merge_detections(np.concatenate(boxes),
                                     np.concatenate(scores))
    keep = scores >= conf
    return boxes[keep], scores[keep]


def choose_input_scale(detector, cap, total, args):
    """A coarse guard against footage shot far closer than the training clips.

    Fires only past AUTO_TRIGGER times the trained median, because the median
    detection runs large on any clip, so an in-range clip measures bigger than
    its own labels. Reads no ground truth.
    """
    sizes = []
    for quantile in (0.2, 0.5, 0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * quantile))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes, _ = detect_frame(detector, frame, args.tile, args.overlap,
                                args.conf, args.batch)
        sizes += [float(max(b[2] - b[0], b[3] - b[1])) for b in boxes]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(sizes) < 5:
        print("  auto scale: too few detections to judge, staying at 1.0")
        return 1.0

    median = sorted(sizes)[len(sizes) // 2]
    trigger = args.train_median_px * AUTO_TRIGGER
    if median <= trigger:
        print(f"  auto scale: median detection {median:.0f}px over "
              f"{len(sizes)} boxes, within {trigger:.0f}px — staying at 1.0")
        return 1.0
    scale = min(SCALE_STEPS, key=lambda s: abs(s - args.train_median_px / median))
    print(f"  auto scale: median detection {median:.0f}px over {len(sizes)} "
          f"boxes, past {trigger:.0f}px -> scale {scale}")
    return scale


def process(detector, video, out_dir, args):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  cannot open {video.name}, skipped")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {width}x{height}, {total} frames at {fps:.0f} fps")

    scale = (choose_input_scale(detector, cap, total, args)
             if str(args.input_scale).lower() == "auto" else float(args.input_scale))

    vis_size = even_size(width, height, args.vis_width)
    writer = None if args.no_video else FragmentedWriter(
        out_dir / f"{video.stem}.mp4", fps / args.frame_stride, vis_size)
    csv_file = open(out_dir / f"{video.stem}.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_s", "conf", "x1", "y1", "x2", "y2"])

    first = int(args.start_s * fps)
    last = first + int(args.seconds * fps) if args.seconds else None
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    started, idx, processed, found = time.time(), first - 1, 0, 0
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
            if scale != 1.0:
                small = cv2.resize(frame, (int(width * scale), int(height * scale)),
                                   interpolation=cv2.INTER_AREA)
                boxes, scores = detect_frame(detector, small, args.tile,
                                             args.overlap, args.conf, args.batch)
                boxes = boxes / scale
            else:
                boxes, scores = detect_frame(detector, frame, args.tile,
                                             args.overlap, args.conf, args.batch)

            for (x1, y1, x2, y2), score in zip(boxes.tolist(), scores.tolist()):
                csv_writer.writerow([idx, round(idx / fps, 3), round(score, 4),
                                     int(x1), int(y1), int(x2), int(y2)])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), max(3, width // 400))
            found += len(boxes)

            if writer is not None:
                if (frame.shape[1], frame.shape[0]) != vis_size:
                    frame = cv2.resize(frame, vis_size, interpolation=cv2.INTER_AREA)
                writer.write(frame)
            processed += 1
            if processed % 25 == 0:
                print(f"    {idx + 1}/{total} frames, {found} detections, "
                      f"{(time.time() - started) / processed:.2f}s/frame",
                      flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()

    print(f"  {processed} frames, {found} detections in {time.time() - started:.0f}s")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/vehicle_detector.onnx")
    ap.add_argument("--in", dest="source", default="videos/in",
                    help="a video file, or a directory of them")
    ap.add_argument("--out", default="videos/out")
    ap.add_argument("--conf", type=float, default=0.50,
                    help="shipping threshold, chosen on the validation clip")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this much footage, for a quick look")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--vis-width", type=int, default=1920)
    ap.add_argument("--train-median-px", type=float, default=TRAIN_MEDIAN_PX)
    ap.add_argument("--input-scale", default="1.0",
                    help='native tiling by default; "auto" shrinks footage '
                         "shot far closer than the training clips, and a "
                         "number applies that scale directly")
    ap.add_argument("--no-video", action="store_true",
                    help="write the csv only")
    ap.add_argument("--threads", type=int, default=0,
                    help="onnxruntime threads, 0 leaves it to the runtime")
    args = ap.parse_args()

    source = Path(args.source)
    if source.is_dir():
        videos = sorted(p for p in source.iterdir()
                        if p.suffix.lower() in SUFFIXES)
    elif source.is_file():
        videos = [source]
    else:
        sys.exit(f"nothing to read at {source}")
    if not videos:
        sys.exit(f"no videos in {source} "
                 f"(looking for {', '.join(sorted(SUFFIXES))})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = OnnxDetector(args.model, threads=args.threads)
    print(f"model {args.model}, tiles {args.tile} px with {args.overlap} px "
          f"overlap, threshold {args.conf}")
    print(f"{len(videos)} video(s) from {source} -> {out_dir}\n")

    for video in videos:
        print(f"{video.name}")
        process(detector, video, out_dir, args)
        print()


if __name__ == "__main__":
    main()
