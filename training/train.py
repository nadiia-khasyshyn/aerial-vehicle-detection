"""Train the sub-500k detector from scratch on the labelled tiles.

No pretrained weights: the custom width/channel scaling means no published
checkpoint fits this architecture, and the domain (top-down road scenes) is
narrow enough that a from-scratch schedule converges on a few thousand tiles.
The pseudo-labels already carry the large models' knowledge, which is the
distillation that matters here.

The parameter budget is asserted before training starts, so an accidental
config change cannot silently ship an oversized model.
"""

import argparse
from pathlib import Path

PARAM_BUDGET = 500_000


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
    ap.add_argument("--resume", action="store_true",
                    help="resume the newest runs/detect/<name>*/ from its last.pt "
                         "(what restart_on_crash.sh passes after a crash)")
    ap.add_argument("--count-only", action="store_true",
                    help="print the parameter count and exit")
    args = ap.parse_args()

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

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # A tiny model from scratch wants the full augmentation schedule and
        # a clean final stretch; mosaic off for the last 20 epochs lets the
        # box regression settle on realistic layouts.
        close_mosaic=20,
        cos_lr=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nbest checkpoint: {best}")
    print("validate on the train-video val split (never the eval clip):")
    print(f"  yolo val model={best} data={args.data} imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
