"""Run the shipped detector over a clip, tiled, and write predictions + video.

Same tiling the training data was cut with (512 px at native resolution), so
the model sees at inference exactly the object scale it was trained on.
Per-tile detections are merged into frame coordinates with class-agnostic NMS;
boxes a tile seam cut in half mostly disappear into the overlap, the rest are
cleaned by the merge.

The confidence threshold is a shipping decision: pick it on the train-video
val split, never on the eval clip's numbers (rule 1).

Outputs:
  <out>/predictions.csv        frame, time_s, conf, x1, y1, x2, y2
  <out>/annotated.mp4          predictions drawn on the clip
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms

from ultralytics import YOLO


class FragmentedWriter:
    """Fragmented MP4 via ffmpeg: an interrupted run still leaves a playable file."""

    def __init__(self, path, fps, size):
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-crf", "23",
             "-g", "30", "-frag_duration", "1000000", "-flush_packets", "1",
             "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
             str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()


def tile_origins(size, tile, stride):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


MERGE_FLOOR = 0.2
MAX_OVERLAP = 0.2   # IoU above which two vehicle boxes are the same vehicle


def detect_frame(model, frame, tile, overlap, conf, device, batch):
    """Tiled detection at the clip's native resolution.

    Train and inference see one object scale: the tiles are cut from the full
    4K frame, exactly as the training tiles were.

    The tiles are read at MERGE_FLOOR and `conf` is applied only after the
    merge. A vehicle split across a seam often scores high on one fragment and
    low on the whole-vehicle box; thresholding first throws the whole box away
    and ships the fragment, which is what put part-boxes in the rendered video
    while the same frame looked right at a lower threshold.
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
                                conf=floor, iou=0.6, verbose=False)
        for res, (ox, oy) in zip(results, offsets[s:s + batch]):
            b = res.boxes
            if len(b) == 0:
                continue
            xyxy = b.xyxy.cpu()
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            boxes.append(xyxy)
            scores.append(b.conf.cpu())

    if not boxes:
        return torch.zeros((0, 4)), torch.zeros(0)
    boxes, scores = merge_detections(torch.cat(boxes), torch.cat(scores))
    keep = scores >= conf
    return boxes[keep], scores[keep]


def merge_detections(boxes, scores):
    """NMS plus containment MERGE (non-max merging, as in SAHI).

    Tile seams leave part-boxes (a hood from one tile, the full car from the
    neighbouring one); their IoU is too small for plain NMS. A box whose own
    area lies >70% inside an already-kept box is folded in: the kept box grows
    to their union and keeps the higher confidence, so the final box covers
    the whole vehicle instead of its best-scoring piece.
    """
    keep = nms(boxes, scores, 0.5)
    boxes, scores = boxes[keep], scores[keep]
    order = scores.argsort(descending=True)
    boxes, scores = boxes[order].clone(), scores[order].clone()
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    kept = []
    for i in range(len(boxes)):
        merged = False
        for j in kept:
            lt = torch.maximum(boxes[i, :2], boxes[j, :2])
            rb = torch.minimum(boxes[i, 2:], boxes[j, 2:])
            inter = (rb - lt).clamp(min=0).prod()
            # intersection over the smaller area: folds a part-box into the
            # whole regardless of which of the two scored higher
            if inter / torch.minimum(areas[i], areas[j]).clamp(min=1e-6) > 0.7:
                boxes[j, :2] = torch.minimum(boxes[j, :2], boxes[i, :2])
                boxes[j, 2:] = torch.maximum(boxes[j, 2:], boxes[i, 2:])
                areas[j] = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
                merged = True
                break
        if not merged:
            kept.append(i)
    idx = torch.tensor(kept, dtype=torch.long)
    boxes, scores = boxes[idx], scores[idx]
    # Seen from above, two vehicles never share pixels: there is no
    # perspective occlusion. Any residual overlap is one vehicle counted
    # twice, so keep the more confident box.
    keep = nms(boxes, scores, MAX_OVERLAP)
    return boxes[keep], scores[keep]


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
    scale = args.vis_width / W if W > args.vis_width else 1.0
    # yuv420p requires even dimensions; a portrait 2160x3840 clip scaled by
    # 1920/2160 lands on an odd height and kills ffmpeg instantly.
    vis_size = (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2)
    writer = None if args.no_video else FragmentedWriter(
        out / "annotated.mp4", fps / args.frame_stride, vis_size)

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
