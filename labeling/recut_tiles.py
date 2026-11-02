"""Rebuild the tile images from the clips, using the label files as the index.

The labels are the work product and they ship in the repo; the images are
derived from public Pexels clips and do not. A tile name carries everything
needed to cut it again — `16186325_f00038_x0512_y0256r90` is clip 16186325,
frame 38, origin (512, 256), rotated 90° — so the dataset can be reconstructed
exactly rather than downloaded.

    bash data/videos/fetch.sh                    # where to get the clips
    python labeling/recut_tiles.py --labels dataset/train/labels \\
        --clips data/videos --out dataset/train/images

Frames are read in one pass per clip, in frame order, because seeking a 4K
video thousands of times is far slower than decoding it once.
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, parse_tile_name   # noqa: E402

TILE = int(get("tiling.tile"))
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def find_clip(clips_dir, clip_id):
    """The clip file whose name starts with this id, whatever the suffix."""
    for path in sorted(Path(clips_dir).iterdir()):
        if path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi") \
                and path.stem.split("-")[0].split("_")[0] == clip_id:
            return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="the split's labels/ dir")
    ap.add_argument("--clips", default="data/videos")
    ap.add_argument("--out", required=True, help="where to write the images")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    wanted = collections.defaultdict(list)      # clip -> [(frame, x, y, rot, stem)]
    unparsed = 0
    for label in sorted(Path(args.labels).glob("*.txt")):
        name = parse_tile_name(label.stem)
        if name is None:
            unparsed += 1
            continue
        rot = 0
        if "r" in label.stem.rsplit("_y", 1)[-1]:
            rot = int(label.stem.rsplit("r", 1)[-1])
        wanted[name.clip].append((name.frame, name.x, name.y, rot, label.stem))
    if unparsed:
        print(f"skipped {unparsed} labels whose names are not tile names")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = missing_clips = 0

    for clip_id in sorted(wanted):
        clip = find_clip(args.clips, clip_id)
        if clip is None:
            print(f"  {clip_id}: clip not found in {args.clips}, skipped")
            missing_clips += 1
            continue
        tiles = sorted(wanted[clip_id])
        by_frame = collections.defaultdict(list)
        for frame, x, y, rot, stem in tiles:
            by_frame[frame].append((x, y, rot, stem))

        cap = cv2.VideoCapture(str(clip))
        index, produced = -1, 0
        remaining = set(by_frame)
        while remaining:
            ok, frame_img = cap.read()
            if not ok:
                break
            index += 1
            if index not in remaining:
                continue
            remaining.discard(index)
            for x, y, rot, stem in by_frame[index]:
                crop = frame_img[y:y + args.tile, x:x + args.tile]
                if crop.shape[:2] != (args.tile, args.tile):
                    continue
                if rot:
                    crop = cv2.rotate(crop, ROTATIONS[rot])
                cv2.imwrite(str(out_dir / f"{stem}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                produced += 1
        cap.release()
        written += produced
        print(f"  {clip_id}: {produced} tiles from {clip.name}"
              + (f", {len(remaining)} frames not reached" if remaining else ""))

    print(f"\n{written} tiles -> {out_dir}")
    if missing_clips:
        sys.exit(f"{missing_clips} clip(s) missing — see data/videos/manifest.csv")


if __name__ == "__main__":
    main()
