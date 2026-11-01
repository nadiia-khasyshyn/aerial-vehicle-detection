"""Tiled YOLO-World detection for aerial traffic footage.

Runs 640x640 sliding-window inference over each frame, merges per-tile
detections back into full-frame coordinates and resolves tile-seam duplicates
with class-agnostic NMS + containment suppression.

Outputs per video:
  output/videos/<name>_annotated.mp4   visualisation with boxes
  output/crops/<name>/<class>/...jpg   detection crops (5 px padding)
  output/detections/<name>.csv         raw detections
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO, YOLOWorld

# Prompts picked from the four clips: top-down and oblique drone views of
# intersections, a highway and a suburban street.
# "... seen from above" roughly doubles recall over the bare nouns: the bare
# prompts are anchored on side-on COCO imagery and miss most of a top-down
# frame. Measured on 3405804 frame 150: 40 detections -> 73.
TARGET_CLASSES = [
    "car seen from above",
    "van seen from above",
    "bus seen from above",
    "truck seen from above",
    "motorcycle seen from above",
    "moped seen from above",
    "scooter seen from above",
    "person seen from above",
]

# Distractors. An open-vocabulary head scores every prompt against the same
# region, so naming what the false positives actually are gives those regions
# a better-fitting home than "bus". These are the FP sources seen in the first
# run: kerbs, lane paint, zebra stripes, rooftops and tree canopy.
DISTRACTOR_CLASSES = [
    "white road marking",
    "white arrow painted on the road",
    "zebra crossing stripes",
    "white lane divider line",
    "kerb",
    "rooftop",
    "building",
    "tree",
    # Low-altitude street footage: a top-down umbrella is a compact coloured
    # blob on asphalt, which the vehicle prompts otherwise claim.
    "umbrella seen from above",
    "pedestrian seen from above",
    "manhole cover",
    "drain grate",
]

CLASSES = TARGET_CLASSES + DISTRACTOR_CLASSES

# Everything except "person" is one `vehicle` class for the labelling task.
VEHICLE_PROMPTS = {c for c in TARGET_CLASSES if "person" not in c}

COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 165, 255), (255, 0, 255), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (128, 0, 255), (255, 255, 255), (0, 128, 255),
]


class FragmentedWriter:
    """Pipe frames to ffmpeg as a fragmented MP4.

    cv2.VideoWriter only writes the moov atom on release, so a run that is
    killed or interrupted mid-video leaves a file no player can open. Frames
    are all there, but nothing can address them. Fragmented MP4 carries its
    index per fragment instead, so a truncated file still plays up to the
    last frame written.
    """

    def __init__(self, path, fps, size):
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-crf", "23",
             # A fragment is flushed per keyframe, so the default 250-frame
             # GOP would still lose everything written since the last one.
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


def load_model(weights):
    """YOLO-World needs its prompts set; a fine-tuned model carries its own."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if "World" in type(ckpt["model"]).__name__:
        model = YOLOWorld(weights)
        model.set_classes(CLASSES)  # must precede the first predict() call
        return model
    return YOLO(weights)


def tile_origins(size, tile, stride):
    """Left/top offsets covering `size`, last tile flush with the far edge."""
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def containment_suppress(boxes, scores, thresh=0.8, min_area_ratio=0.3):
    """Drop a box when most of it sits inside a comparably sized, higher-scoring box.

    NMS alone leaves the partial boxes that tile seams cut out of an object:
    a half-truck against the whole truck can land well under the IoU
    threshold while clearly being the same vehicle.

    `min_area_ratio` is what keeps that from inverting: without it a big
    spurious box along a kerb swallows every real car inside it, and lowering
    the confidence threshold then *lowers* the detection count.
    """
    order = torch.argsort(scores, descending=True)
    keep = []
    for i in order.tolist():
        bi = boxes[i]
        area_i = max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1e-6)
        dropped = False
        for j in keep:
            bj = boxes[j]
            area_j = max((bj[2] - bj[0]) * (bj[3] - bj[1]), 1e-6)
            if area_i / area_j < min_area_ratio:
                continue  # a small box inside a big one is a different object
            ix1 = torch.maximum(bi[0], bj[0])
            iy1 = torch.maximum(bi[1], bj[1])
            ix2 = torch.minimum(bi[2], bj[2])
            iy2 = torch.minimum(bi[3], bj[3])
            inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
            if inter / area_i > thresh:
                dropped = True
                break
        if not dropped:
            keep.append(i)
    return torch.tensor(keep, dtype=torch.long)


def merge_detections(boxes, scores, labels, iou=0.5, contain=0.8):
    if len(boxes) == 0:
        return boxes, scores, labels
    # Class-agnostic: "car"/"van"/"truck" fire on the same vehicle and the
    # per-class NMS inside the model never sees across those prompts.
    keep = nms(boxes, scores, iou)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = containment_suppress(boxes, scores, contain)
    return boxes[keep], scores[keep], labels[keep]


def detect_frame(model, frame, tile, stride, conf, device, batch_size,
                 max_area_frac, max_aspect=5.0, distractor_ids=frozenset(),
                 imgsz=None):
    h, w = frame.shape[:2]
    tiles, offsets = [], []
    for y in tile_origins(h, tile, stride):
        for x in tile_origins(w, tile, stride):
            tiles.append(frame[y:y + tile, x:x + tile])
            offsets.append((x, y))

    boxes, scores, labels = [], [], []
    for s in range(0, len(tiles), batch_size):
        chunk = tiles[s:s + batch_size]
        results = model.predict(chunk, device=device, imgsz=imgsz or tile, conf=conf,
                                iou=0.6, verbose=False)
        for res, (ox, oy) in zip(results, offsets[s:s + batch_size]):
            b = res.boxes
            if len(b) == 0:
                continue
            xyxy = b.xyxy.cpu()
            # A box swallowing the tile is the model latching onto the
            # background (road surface, tree canopy), not an object. The
            # aspect guard drops the long thin boxes it draws along kerbs and
            # lane markings; a vehicle from above tops out near 4:1.
            bw = xyxy[:, 2] - xyxy[:, 0]
            bh = xyxy[:, 3] - xyxy[:, 1]
            aspect = torch.maximum(bw / bh.clamp(min=1), bh / bw.clamp(min=1))
            cls = b.cls.cpu().long()
            ok = (bw * bh < max_area_frac * tile * tile) & (aspect < max_aspect)
            if distractor_ids:
                # Both guards describe what a vehicle looks like, so they
                # delete the distractors on sight: lane paint is long and
                # thin, a building covers half the tile. Exempt them.
                is_dis = torch.tensor([c.item() in distractor_ids for c in cls],
                                      dtype=torch.bool)
                ok = ok | is_dis
            if not ok.any():
                continue
            xyxy = xyxy[ok]
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            boxes.append(xyxy)
            scores.append(b.conf.cpu()[ok])
            labels.append(cls[ok])

    if not boxes:
        empty = torch.zeros((0, 4))
        return empty, torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return merge_detections(torch.cat(boxes), torch.cat(scores), torch.cat(labels))


def draw(frame, boxes, scores, labels, names, thickness, font_scale):
    for (x1, y1, x2, y2), sc, cl in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        color = COLORS[int(cl) % len(COLORS)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, thickness)
        text = f"{names[int(cl)]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def save_crops(frame, boxes, labels, names, out_dir, frame_idx, pad):
    h, w = frame.shape[:2]
    for i, ((x1, y1, x2, y2), cl) in enumerate(zip(boxes.tolist(), labels.tolist())):
        x1 = max(int(x1) - pad, 0)
        y1 = max(int(y1) - pad, 0)
        x2 = min(int(x2) + pad, w)
        y2 = min(int(y2) + pad, h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cls_dir = out_dir / names[int(cl)].replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cls_dir / f"f{frame_idx:05d}_{i:03d}.jpg"),
                    frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_video(model, path, out_root, args):
    name = path.stem.split("-")[0]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_dir = out_root / "videos"
    crop_dir = out_root / "crops" / name
    csv_dir = out_root / "detections"
    for d in (vid_dir, crop_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    scale = args.vis_width / w if args.vis_width and w > args.vis_width else 1.0
    # yuv420p needs even dimensions; a portrait clip scaled to vis_width lands
    # on an odd height and ffmpeg dies on the first frame.
    vw, vh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    writer = FragmentedWriter(vid_dir / f"{name}_annotated.mp4",
                              fps / args.frame_stride, (vw, vh))

    csv_f = open(csv_dir / f"{name}.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "time_s", "class", "conf", "x1", "y1", "x2", "y2"])

    try:
        names = model.model.names
        distractor_ids = set() if isinstance(names, dict) else {
            i for i, n in enumerate(names) if n in DISTRACTOR_CLASSES}
        counts = {}
        n_proc = 0
        t0 = time.time()
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.frame_stride:
                continue

            boxes, scores, labels = detect_frame(
                model, frame, args.tile, args.tile - args.overlap, args.conf,
                args.device, args.batch, args.max_area_frac, args.max_aspect,
            distractor_ids, args.imgsz)

            for (x1, y1, x2, y2), sc, cl in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                cname = names[int(cl)]
                counts[cname] = counts.get(cname, 0) + 1
                csv_w.writerow([idx, round(idx / fps, 3), cname, round(sc, 4),
                                int(x1), int(y1), int(x2), int(y2)])

            if n_proc % args.crop_every == 0:
                save_crops(frame, boxes, labels, names, crop_dir, idx, args.pad)

            vis = draw(frame, boxes, scores, labels, names,
                       max(2, int(3 * scale * 2)), 0.9)
            if (vis.shape[1], vis.shape[0]) != (vw, vh):
                vis = cv2.resize(vis, (vw, vh), interpolation=cv2.INTER_AREA)
            writer.write(vis)

            n_proc += 1
            if n_proc % 10 == 0:
                el = time.time() - t0
                print(f"  [{name}] {idx + 1}/{total} frames  "
                      f"{el / n_proc:.2f}s/frame  {sum(counts.values())} dets", flush=True)

    finally:
        cap.release()
        writer.release()
        csv_f.close()
    print(f"  [{name}] done: {n_proc} frames in {time.time() - t0:.0f}s")
    print(f"  [{name}] detections: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    return name, n_proc, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--weights", default="yolov8x-worldv2.pt")
    p.add_argument("--out", default="output")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=None,
                   help="inference size; defaults to --tile. Set it for "
                        "full-frame mode (--tile larger than the frame), where "
                        "feeding the raw 4K frame to the model is not viable")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--max-area-frac", type=float, default=0.15)
    p.add_argument("--max-aspect", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--crop-every", type=int, default=5)
    p.add_argument("--pad", type=int, default=5)
    p.add_argument("--vis-width", type=int, default=1920)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    root = Path.cwd()
    videos = [Path(v) for v in args.videos] if args.videos else sorted(root.glob("*.mp4"))
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = root / out_root

    model = load_model(args.weights)
    # set_classes leaves names as a list; a fine-tuned model carries a dict.
    raw = model.model.names
    print(f"classes: {list(raw.values()) if isinstance(raw, dict) else list(raw)}")
    print(f"tiling: {args.tile}px, stride {args.tile - args.overlap}, conf {args.conf}")

    summary = []
    for v in videos:
        print(f"\n== {v.name}")
        summary.append(process_video(model, v, out_root, args))

    print("\n=== summary ===")
    for name, n, counts in summary:
        print(f"{name}: {n} frames, {sum(counts.values())} detections, {counts}")


if __name__ == "__main__":
    main()
