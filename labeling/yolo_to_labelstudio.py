"""Build a Label Studio project from a YOLO dataset: tiles + current labels
as pre-annotations, so review means correcting boxes instead of drawing them.

Run from the workspace directory (like every labeling script). Requires a
running Label Studio with local file serving enabled:

    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD
    label-studio start

Then:

    python yolo_to_labelstudio.py --images dataset/train/images \
        --labels dataset/train/labels --title "vehicle tiles review" \
        --token <API token from Account & Settings>

The exported corrections are compared against the originals with
labelstudio_diff.py to get the added / removed / fixed counts for the README.
"""

import argparse
import json
from pathlib import Path

import requests
from PIL import Image

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="vehicle" background="#00cc44"/>
  </RectangleLabels>
</View>
"""


def yolo_boxes(label_path):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            _, cx, cy, w, h = (float(v) for v in parts[:5])
            boxes.append((cx, cy, w, h))
    return boxes


def task_for(image_path, label_path, root):
    with Image.open(image_path) as im:
        width, height = im.size
    results = []
    for cx, cy, w, h in yolo_boxes(label_path):
        results.append({
            "type": "rectanglelabels",
            "from_name": "label", "to_name": "image",
            "original_width": width, "original_height": height,
            "value": {
                "x": (cx - w / 2) * 100, "y": (cy - h / 2) * 100,
                "width": w * 100, "height": h * 100,
                "rotation": 0, "rectanglelabels": ["vehicle"],
            },
        })
    rel = image_path.resolve().relative_to(root.resolve())
    return {
        "data": {"image": f"/data/local-files/?d={rel}"},
        "predictions": [{"model_version": "pseudo-labels", "result": results}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--title", default="vehicle tiles review")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--token", required=True, help="Label Studio API token")
    ap.add_argument("--root", default=".",
                    help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (default: cwd)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N tiles (0 = all)")
    args = ap.parse_args()

    headers = {"Authorization": f"Token {args.token}"}
    root = Path(args.root)
    images = sorted(Path(args.images).glob("*.jpg")) + \
        sorted(Path(args.images).glob("*.png"))
    if args.limit:
        images = images[:args.limit]
    labels_dir = Path(args.labels)

    resp = requests.post(f"{args.url}/api/projects", headers=headers, json={
        "title": args.title, "label_config": LABEL_CONFIG.strip()})
    resp.raise_for_status()
    project = resp.json()["id"]
    print(f"project {project}: {args.title}")

    # LS serves a local file only when its directory belongs to a Local Files
    # storage registered in the project; env vars alone give 404s. Tiles may be
    # symlinks, so register the resolved parent dirs.
    roots = sorted({str(p.resolve().parent) for p in images})
    for root_dir in roots:
        resp = requests.post(f"{args.url}/api/storages/localfiles",
                             headers=headers, json={
                                 "project": project, "path": root_dir,
                                 "title": root_dir, "use_blob_urls": False})
        resp.raise_for_status()
        print(f"storage: {root_dir}")

    tasks = [task_for(p, labels_dir / f"{p.stem}.txt", root) for p in images]
    resp = requests.post(f"{args.url}/api/projects/{project}/import",
                         headers=headers, json=tasks)
    resp.raise_for_status()
    n_boxes = sum(len(t["predictions"][0]["result"]) for t in tasks)
    print(f"imported {len(tasks)} tiles, {n_boxes} pre-annotated boxes")
    print(f"open: {args.url}/projects/{project}")


if __name__ == "__main__":
    main()
