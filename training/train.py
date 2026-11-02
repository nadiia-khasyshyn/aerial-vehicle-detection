"""Train the sub-500k detector on the labelled tiles.

No published checkpoint fits this scaling, so there are two pretrain routes:
--pretrain-coco trains this same architecture on COCO's vehicle classes and
then fine-tunes on the tiles (everything but the final class convolutions
transfers), while --pretrained <ckpt> copies whatever matches by shape from any
checkpoint. The parameter budget is checked before training and on the result.
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get   # noqa: E402

PARAM_BUDGET = int(get("model.param_budget"))
TRAIN_CFG = get("training")


def resolve_device(choice):
    if choice != "auto":
        return choice
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def heal_data_yaml(data_path):
    """Point data.yaml's `path` at its own directory when the stored one is stale.

    make_split.py writes an absolute path, which breaks the moment the dataset
    is rsynced to a pod or another machine; the yaml always sits inside the
    dataset root, so that root is the one path that is always right.
    """
    import yaml
    data_path = Path(data_path)
    config = yaml.safe_load(open(data_path))
    root = str(data_path.parent.resolve())
    if config.get("path") != root and not Path(config.get("path", "")).exists():
        config["path"] = root
        data_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"data.yaml path updated -> {root}")


def find_last_checkpoint(name):
    candidates = sorted(Path("runs").glob(f"detect/{name}*/weights/last.pt"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"--resume: no runs/detect/{name}*/weights/last.pt found")
    return candidates[-1]


def check_budget(weights):
    """Count the trained checkpoint: a pretrain can widen the head past the yaml."""
    from ultralytics import YOLO

    n_params = sum(p.numel() for p in YOLO(str(weights)).model.parameters())
    print(f"trained checkpoint: {n_params:,} parameters (budget {PARAM_BUDGET:,})")
    if n_params >= PARAM_BUDGET:
        raise SystemExit(f"{weights} exceeds the parameter budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).parent.parent
                                           / "configs" / "vehicle_tiny_p34.yaml"))
    ap.add_argument("--data", default="dataset_final/data.yaml")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="matches the tile size the model sees at inference")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto",
                    help="auto picks cuda, then mps, then cpu")
    ap.add_argument("--name", default="vehicle_tiny")
    ap.add_argument("--pretrained", default=None,
                    help="checkpoint to transfer shape-matching weights from "
                         "(a path, or a name like yolov8n.pt that ultralytics downloads)")
    ap.add_argument("--pretrain-coco", action="store_true",
                    help="stage 1: train this same architecture on COCO vehicle "
                         "classes, then fine-tune on --data with those weights "
                         "(downloads COCO ~20GB; run it on the pod)")
    ap.add_argument("--coco-epochs", type=int, default=30)
    ap.add_argument("--resume", action="store_true",
                    help="resume the newest runs/detect/<name>*/ from its last.pt "
                         "(what restart_on_crash.sh passes after a crash)")
    ap.add_argument("--count-only", action="store_true",
                    help="print the parameter count and exit")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="extra ultralytics train() overrides, e.g. "
                         "--set flipud=0.5 degrees=30 (top-down footage has "
                         "no gravity direction, unlike the defaults)")
    args = ap.parse_args()
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            overrides[key] = float(value)
        except ValueError:
            overrides[key] = value

    from ultralytics import YOLO

    if args.resume:
        last = find_last_checkpoint(args.name)
        print(f"resuming {last}")
        model = YOLO(last)
        model.train(resume=True)
        return

    model = YOLO(args.model)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"parameters: {n_params:,} (budget {PARAM_BUDGET:,})")
    assert n_params < PARAM_BUDGET, "model exceeds the parameter budget"
    if args.count_only:
        return

    heal_data_yaml(args.data)
    args.device = resolve_device(args.device)
    print(f"device: {args.device}")

    if args.pretrain_coco:
        # COCO (~37GB) lands in ultralytics' datasets_dir (~/datasets), which
        # settings.update() cannot move at runtime — bootstrap.sh symlinks it to
        # the pod volume. Locally ~/datasets needs ~40GB free.
        # Ids 1,2,3,5,7: bicycle, car, motorcycle, bus, truck.
        print(f"stage 1/2: COCO vehicles pretrain, {args.coco_epochs} epochs "
              f"(note: a crash-restart --resume resumes only the stage it lands in)")
        stage = YOLO(args.model)
        stage.train(data="coco.yaml", classes=[1, 2, 3, 5, 7],
                    imgsz=args.imgsz, epochs=args.coco_epochs, batch=args.batch,
                    device=args.device, name=f"{args.name}_coco", cos_lr=True)
        args.pretrained = str(Path(stage.trainer.save_dir) / "weights" / "best.pt")
        print(f"stage 2/2: fine-tune from {args.pretrained}")

    if args.pretrained:
        model.load(args.pretrained)  # prints how many tensors transferred

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # Mosaic off for the last 20 epochs: box regression settles on
        # realistic layouts rather than stitched ones.
        close_mosaic=20,
        cos_lr=True,
        **overrides,
    )
    check_budget(model.trainer.best)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nbest checkpoint: {best}")
    print("validate on the train-video val split (never the eval clip):")
    print(f"  yolo val model={best} data={args.data} imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
