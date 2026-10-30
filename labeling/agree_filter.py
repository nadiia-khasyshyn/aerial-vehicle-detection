"""Keep only the boxes two independent detectors agree on.

The custom model and YOLO-World fail in different places: the custom one is
blind to scooters and fires on zebra stripes, YOLO-World sees two-wheelers but
scatters low-confidence noise over kerbs and rooftops. Their errors barely
overlap, so agreement is a strong proxy for a real vehicle — far stronger than
either model's own confidence.

Runs over the already-cut 512 px tiles and writes YOLO labels for the
intersection, plus a visualisation showing what each model contributed.
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou

from ultralytics import YOLO, YOLOWorld

# The set that measured 83 detections on 3405804 frame 150. Adding moped,
# scooter and the distractor prompts diluted the scores and lost the two most
# obvious cars, so this stays as it was validated.
WORLD_PROMPTS = [
    "car seen from above",
    "van seen from above",
    "bus seen from above",
    "truck seen from above",
    "motorcycle seen from above",
]


def run_model(model, paths, imgsz, conf, device, batch):
    """path -> (boxes, scores) in tile coordinates."""
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        results = model.predict(imgs, device=device, imgsz=imgsz, conf=conf,
                                iou=0.6, verbose=False)
        for p, res in zip(chunk, results):
            b = res.boxes
            out[p] = (b.xyxy.cpu(), b.conf.cpu()) if len(b) else \
                     (torch.zeros((0, 4)), torch.zeros(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_512/images")
    ap.add_argument("--out", default="dataset_512_agree")
    ap.add_argument("--custom", default="weights/labeler_custom.pt")
    ap.add_argument("--world", required=True)
    ap.add_argument("--conf-custom", type=float, default=0.08)
    ap.add_argument("--conf-world", type=float, default=0.05)
    ap.add_argument("--imgsz-custom", type=int, default=512)
    ap.add_argument("--imgsz-world", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    paths = sorted((root / args.images).glob("*.jpg"))
    out = root / args.out
    vis = root / f"{args.out}_vis"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} tiles", flush=True)

    t0 = time.time()
    custom = YOLO(args.custom)
    det_c = run_model(custom, paths, args.imgsz_custom, args.conf_custom,
                      args.device, args.batch)
    print(f"custom done in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    world = YOLOWorld(args.world)
    world.set_classes(WORLD_PROMPTS)  # must precede the first predict
    det_w = run_model(world, paths, args.imgsz_world, args.conf_world,
                      args.device, args.batch)
    print(f"world done in {time.time() - t0:.0f}s", flush=True)

    n_c = n_w = n_a = 0
    tiles_kept = 0
    for p in paths:
        bc, sc = det_c[p]
        bw, _ = det_w[p]
        n_c += len(bc)
        n_w += len(bw)

        if len(bc) and len(bw):
            # Class-agnostic on purpose: the two label spaces do not line up
            # ("other_vehicle" against five prompts), and the question is only
            # whether both put an object here.
            hit = (box_iou(bc, bw) >= args.iou).any(dim=1)
            keep = bc[hit]
            keep_s = sc[hit]
        else:
            keep = torch.zeros((0, 4))
            keep_s = torch.zeros(0)
        n_a += len(keep)

        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        with open(out / "labels" / f"{p.stem}.txt", "w") as fh:
            for x1, y1, x2, y2 in keep.tolist():
                fh.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        if len(keep):
            tiles_kept += 1

        shown = img.copy()
        for x1, y1, x2, y2 in bc.tolist():      # custom only: dropped
            cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        for x1, y1, x2, y2 in bw.tolist():      # world only: dropped
            cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 1)
        for x1, y1, x2, y2 in keep.tolist():    # agreed: kept
            cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.imwrite(str(vis / f"{p.stem}.jpg"), shown, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(out / "data.yaml", "w") as fh:
        fh.write(f"path: {out}\ntrain: {root / args.images}\nval: {root / args.images}\n"
                 f"nc: 1\nnames:\n  0: vehicle\n")

    print(f"\ncustom  @{args.conf_custom}: {n_c} boxes")
    print(f"world   @{args.conf_world}: {n_w} boxes")
    print(f"agreed  @IoU{args.iou}: {n_a} boxes "
          f"({n_a / max(n_c, 1) * 100:.1f}% of custom, "
          f"{n_a / max(n_w, 1) * 100:.1f}% of world)")
    print(f"tiles with at least one agreed box: {tiles_kept}/{len(paths)}")
    print(f"labels: {out}/labels\nvisualisation: {vis}  "
          f"(green kept, red custom-only, orange world-only)")


if __name__ == "__main__":
    main()
