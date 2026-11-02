# Data

The clips themselves are not in the repo: GitHub rejects files over 100 MB and
these are 14–389 MB each. `videos/manifest.csv` lists every clip with its
Pexels id, and `videos/fetch.sh` prints the pages to download them from.
Drop the downloaded files into `videos/` and every script below finds them.

## Clips

| Clip | Role | Scene |
|------|------|-------|
| 8968356 | train | highway interchange (task clip A) |
| 5382494 | train | rural highway, light traffic (task clip B) |
| 8457857 | train | simple highway, top-down (task clip C) |
| 3405804 | train | urban intersection (task clip D) |
| 15857456 | train | added clip, wide highway from altitude |
| 14710575 | val | held-out clip: checkpoint and threshold are chosen here |
| 13722965 | eval | low-altitude street, measured only, never decided on |

The task's own eval clip (32179597) could not be downloaded, so 13722965
stands in for it — see the README's Results section.

## From clips to the training set

```
# 1. propose boxes with an open-vocabulary detector
python labeling/tiled_yoloworld.py --videos data/videos/8968356*.mp4 \
    --tile 640 --overlap 128 --out output_8968356

# 2. cut 512 px tiles at native resolution, with background tiles
python labeling/build_dataset.py --video data/videos/8968356*.mp4 \
    --detections output_8968356/detections/*.csv --out dataset_main

# 3. assemble the split (clip-level: one whole clip is the validation set)
python training/build_dataset.py \
    --train dataset_main/images:dataset_main/labels \
            dataset_bg/images:dataset_bg/labels \
    --val   dataset_val2/images:dataset_val2/labels \
            dataset_val2_bg/images:dataset_val2_bg/labels \
    --out dataset

# 4. review in Label Studio and pull the corrections into the split
python labeling/yolo_to_labelstudio.py --images dataset/train/images \
    --labels dataset/train/labels --title "vehicle tiles" --token $LS_TOKEN
python labeling/labelstudio_to_yolo.py --export export.json \
    --labels dataset/train/labels
python labeling/labelstudio_diff.py --export export.json \
    --labels dataset_main/labels     # diff against the pristine originals

# 5. drop sliver boxes the tiling left behind
python labeling/filter_slivers.py --labels dataset/train/labels --min-side 7
```

The review writes real files over the symlinks, so the untouched originals
stay in `dataset_main/` for the correction tally. Order matters: assemble
first, correct second, otherwise the diff has nothing to compare against.

`dataset/` ends up as `train/{images,labels}` and `val/{images,labels}` plus a
`data.yaml`, with tiles symlinked to the labelled folders. It is gitignored:
2800 + 698 tiles are 330 MB, and they are reproducible from the clips with the
commands above. Pass `--copy` to materialise real files before uploading it to
a rented GPU.

## What the final dataset holds

| Split | Tiles | Boxes | Background tiles | Clips |
|-------|-------|-------|------------------|-------|
| train | 2800 | 8977 | 700 (25%) | 8968356, 5382494, 8457857, 3405804, 15857456 |
| val | 698 | 747 | 200 (29%) | 14710575 |

Background tiles carry no boxes on purpose: without empty road, roof and
kerb tiles the detector learns that every patch of asphalt holds a vehicle.
