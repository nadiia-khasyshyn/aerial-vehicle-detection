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


def detect_frame(model, frame, tile, overlap, conf, device, batch):
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
                                conf=conf, iou=0.6, verbose=False)
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
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    keep = nms(boxes, scores, 0.5)
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
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vis-width", type=int, default=1920)
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
    writer = FragmentedWriter(out / "annotated.mp4", fps / args.frame_stride,
                              (int(W * scale), int(H * scale)))

    csv_f = open(out / "predictions.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "time_s", "conf", "x1", "y1", "x2", "y2"])

    t0 = time.time()
    idx, n_proc, n_det = -1, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
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
            if scale != 1.0:
                frame = cv2.resize(frame, (int(W * scale), int(H * scale)),
                                   interpolation=cv2.INTER_AREA)
            writer.write(frame)
            n_proc += 1
            if n_proc % 10 == 0:
                print(f"  {idx + 1}/{total} frames  "
                      f"{(time.time() - t0) / n_proc:.2f}s/frame  {n_det} dets",
                      flush=True)
    finally:
        cap.release()
        writer.release()
        csv_f.close()

    print(f"\n{n_proc} frames, {n_det} detections in {time.time() - t0:.0f}s")
    print(f"-> {out / 'predictions.csv'}\n-> {out / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
