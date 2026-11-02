# Training and evaluation, step by step

The commands behind the results, for reproducing them from the clips.

## Training

```
python training/train.py --data <dataset>/data.yaml --pretrain-coco  # 2 stages
python training/train.py --data <dataset>/data.yaml                  # scratch
```

imgsz 512 (= tile size), 250 epochs with cosine LR, mosaic off for the last
20 epochs; the parameter budget is asserted before every run. `--pretrain-coco`
runs the COCO-vehicles pretrain described above first (≈20 GB COCO download —
do it on the pod); `--pretrained <ckpt>` instead transfers whatever matches by
shape from any checkpoint. The shipping checkpoint and confidence threshold
are picked on the validation split only.

Training runs locally (`--device auto` picks CUDA, then Apple MPS, then CPU)
or on a rented RunPod GPU via `cloud/` — a slim port of a production cloud
runner, kept to what this task needs (rsync code + dataset up, detached task
that survives dropped SSH, crash-restart from the last checkpoint, artifacts
back to `runs/cloud/<run_id>`; no S3/MLflow):

```
python cloud/train_remote.py --list-gpus                      # prices & stock
python cloud/train_remote.py --data dataset_final/data.yaml \
    --gpu rtx-3090 --estimate-only                            # plan & cost
python cloud/train_remote.py --data dataset_final/data.yaml --gpu rtx-3090
python cloud/train_remote.py --where local                    # laptop mode
```

Needs `RUNPOD_API_KEY` (env or `.env`) and an ed25519 SSH key; pod settings
live in `configs/runpod.yaml`. Ctrl+C detaches without killing the run.

## Eval

```
# 1. predictions over the full confidence range (AP needs it), at the same
#    frame stride the ground truth is annotated on
python eval/predict_video.py --video <clip.mp4> --weights weights/vehicle_detector.pt \
    --conf 0.05 --frame-stride 6 --out eval_run
# 2. ground truth: propose with both allowed detectors and keep what they
#    agree on, then correct the result by hand (csv: frame,x1,y1,x2,y2)
#    python labeling/tiled_yoloworld.py --videos <clip.mp4> --frame-stride 6 \
#        --weights yolov8x-worldv2.pt --out ow
#    python labeling/tiled_yoloworld.py --videos <clip.mp4> --frame-stride 6 \
#        --weights yolov8x.pt --out coco
#    python labeling/agree_frames.py --a ow/detections/<clip>.csv \
#        --b coco/detections/<clip>.csv --out gt_<clip>.csv
# 3. where the ground truth is built from tiles, restrict the predictions to
#    the area those tiles cover — see below, this is not optional
python eval/restrict_to_tiles.py --pred eval_run/predictions.csv \
    --tiles dataset/val/images --clip 14710575 \
    --out eval_run/predictions_in_tiles.csv
# 4. banded metrics; --conf is the shipping threshold chosen on training data
python eval/eval_bands.py --gt gt_val.csv --pred eval_run/predictions_in_tiles.csv \
    --img-width 3840 --fps 25 --conf 0.50
```

**Step 3 is load-bearing.** Validation ground truth comes from tiles, which
cover about 16% of each frame and move from frame to frame, so a detector run
over the whole frame is mostly looking at ground nobody labelled. Counting
those boxes against it measures labelling coverage, not the detector: on the
validation clip the unrestricted file gives 1233 false positives and 0.25 mAP
where the restricted one gives 69 and 0.66. A box is kept when its centre lies
in a tile **belonging to that frame**; using the union of all tiles instead
keeps five times too many.

`eval/eval_bands.py --frames` controls which frames are scored. The default,
`gt`, scores only annotated frames and says how many predictions it ignored;
`union` also counts predictions on unannotated frames as false alarms, which is
right only when every frame carries ground truth.

TP/FN land in the band of the GT box, FP in the band of the prediction's own
estimated distance. Time to first detection links GT boxes into tracks by IoU
across frames (no identity in the csv), then measures first-appearance to
first-match per track; the script also reports how many tracks were never
detected, which the median alone would hide.

Band coverage: the stand-in eval clip is shot from about 15 m and has
nothing beyond 200 m, so its far band is empty and reported as such. The
200–400 m band is covered by the added clip 15857456 and measured on the
validation clip, which spans 35–420 m.

## How `--input-scale auto` decides

It probes three frames at the shipping threshold and compares the median
detection to `inference.train_median_px`, rescaling only past
`inference.auto_scale_trigger` times that size. The margin is deliberate: the
median detection runs large on any clip, because recall rises with box size, so
a clip already in range measures bigger than its own labels and a tighter rule
would shrink footage that needed nothing.

Measured with the shipped weights, trigger at 3 x 28 = 84 px:

| Clip | median detection | chosen |
|---|---|---|
| validation 14710575 | 39 px | 1.0 |
| training 3405804 | 83 px | 1.0 |
| eval 13722965 (15 m) | 121 px | 0.25 |

3405804 clears the trigger by a single pixel, so `auto` separates these three
but with no margin to spare. For a batch of unfamiliar footage the explicit
value is the dependable control.

## Repo layout

`avd/` shared package (config, tile naming, box geometry, ONNX backend) ·
`configs/` every threshold in `pipeline.yaml` · `labeling/` propose → score →
correct → review → export · `training/` dataset assembly, training, ONNX
export · `cloud/` RunPod launcher · `eval/` tiled inference on both backends,
distance model, banded metrics · `docker/` runtime image · `weights/` the
shipped checkpoint and its export.

Two entry points detect and they agree: `eval/detect_onnx.py` is deployment,
`eval/predict_video.py` development. Over 60 validation tiles both return 99
boxes with identical confidences and at most 0.96 px of disagreement; over a
whole clip, the same count on every frame, never more than 2 px apart.

