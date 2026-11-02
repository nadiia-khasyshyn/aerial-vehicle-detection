"""Export the shipped checkpoint to ONNX and check the two agree.

The container runs on onnxruntime alone — no torch, no ultralytics — so the
export is the boundary between training and deployment, and it is worth
verifying rather than trusting. The check runs both the checkpoint and the
exported graph over real tiles and compares the boxes they produce; anything
above a pixel of disagreement means the export is not the model.

    python training/export_onnx.py --weights weights/vehicle_detector.pt \
        --out weights/vehicle_detector.onnx --check dataset/val/images
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get   # noqa: E402

TILE = int(get("tiling.tile"))


def parity_check(pt_path, onnx_path, images, tile, n, conf):
    """Run both backends over the same tiles, report the largest disagreement."""
    import cv2
    from ultralytics import YOLO
    from avd.onnx_detector import OnnxDetector

    paths = sorted(Path(images).glob("*.jpg"))[:n]
    if not paths:
        print(f"check: no tiles in {images}, skipped")
        return True

    torch_model = YOLO(str(pt_path))
    onnx_model = OnnxDetector(onnx_path)

    worst_box, worst_conf, counts = 0.0, 0.0, []
    for path in paths:
        img = cv2.imread(str(path))
        res = torch_model.predict(img, imgsz=tile, conf=conf,
                                  iou=onnx_model.nms_iou, verbose=False)[0]
        a_boxes = res.boxes.xyxy.cpu().numpy()
        a_scores = res.boxes.conf.cpu().numpy()
        b_boxes, b_scores = onnx_model.detect_tile(img, conf)
        counts.append((len(a_boxes), len(b_boxes)))
        if len(a_boxes) != len(b_boxes):
            continue
        if len(a_boxes) == 0:
            continue
        order_a = np.lexsort((a_boxes[:, 1], a_boxes[:, 0]))
        order_b = np.lexsort((b_boxes[:, 1], b_boxes[:, 0]))
        worst_box = max(worst_box,
                        float(np.abs(a_boxes[order_a] - b_boxes[order_b]).max()))
        worst_conf = max(worst_conf,
                         float(np.abs(a_scores[order_a] - b_scores[order_b]).max()))

    mismatched = [c for c in counts if c[0] != c[1]]
    print(f"check: {len(paths)} tiles, "
          f"{sum(c[0] for c in counts)} boxes from the checkpoint, "
          f"{sum(c[1] for c in counts)} from ONNX")
    print(f"  largest box disagreement:        {worst_box:.3f} px")
    print(f"  largest confidence disagreement: {worst_conf:.4f}")
    if mismatched:
        print(f"  tiles where the box count differs: {len(mismatched)}")
    return not mismatched and worst_box < 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/vehicle_detector.pt")
    ap.add_argument("--out", default="weights/vehicle_detector.onnx")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--check", default=None,
                    help="directory of tiles to compare the two backends on")
    ap.add_argument("--check-n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    produced = Path(model.export(format="onnx", imgsz=args.tile,
                                 opset=args.opset, dynamic=True, simplify=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out.resolve():
        produced.replace(out)

    size_mb = out.stat().st_size / 1e6
    print(f"\n{args.weights} ({n_params:,} parameters) -> {out} ({size_mb:.1f} MB)")
    print(f"input 1x3x{args.tile}x{args.tile} (batch axis dynamic), opset {args.opset}")

    if args.check:
        ok = parity_check(args.weights, out, args.check, args.tile,
                          args.check_n, args.conf)
        print("  backends agree" if ok else "  BACKENDS DISAGREE — do not ship this")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
