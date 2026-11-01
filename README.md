# Aerial vehicle detection — ML Engineer test task

A <500k-parameter single-class vehicle detector for drone footage, trained on
model-generated labels from Pexels clips and measured on a held-out eval clip
in two distance bands (0–200 m, 200–400 m).

> ⟨TODO⟩ marks every number that comes from an actual run. Fill them in before
> submitting; nothing below should ship as a placeholder.

## Results

Eval clip: [pexels.com/video/32179597](https://www.pexels.com/video/32179597/)
(city highway, daytime). Shipping confidence threshold **⟨TODO⟩**, chosen on
the train-video validation split, never on these numbers.

| Metric                              | 0–200 m | 200–400 m |
|-------------------------------------|---------|-----------|
| Detection rate TP / (TP + FN)       | ⟨TODO⟩  | ⟨TODO⟩    |
| Precision TP / (TP + FP)            | ⟨TODO⟩  | ⟨TODO⟩    |
| False alarms FP×60/N_frames         | ⟨TODO⟩  | ⟨TODO⟩    |
| False alarms / min of footage       | ⟨TODO⟩  | ⟨TODO⟩    |
| Time to first detection (median, s) | ⟨TODO⟩  | ⟨TODO⟩    |
| mAP@0.5                             | ⟨TODO⟩  | ⟨TODO⟩    |

mAP@0.5 over both bands (bonus): **⟨TODO⟩**. The false-alarm row uses the
task sheet's formula (FP×60/N_frames — alarms per 60 frames); the row under it
is per minute of real footage, which at 30 fps is 30× larger. Both come from
the same run, pick whichever basis you trust.

Final model: **388,786 parameters** (`training/train.py --count-only` prints
and asserts it). Example frames with GT vs predictions are in `examples/`,
the annotated eval video is at ⟨TODO: link⟩, weights at ⟨TODO: link or path⟩.

### Distance model and its assumptions

Pinhole: `d = f_px · L / p`, with `f_px = W / (2·tan(hfov/2))`, `p` the box's
long side in pixels. Assumed: horizontal FOV **84°** (typical consumer drone
main camera), nominal vehicle length **L = 4.5 m**, box long side ≈ vehicle
length. `d` is slant range. Trucks/buses therefore read closer than they are,
and oblique views slightly overestimate distance; both effects and the chosen
constants are in `eval/distance.py`, change them there to disagree.

## The model (rule 5)

YOLOv8 layout cut down twice (`configs/vehicle_tiny_p34.yaml`):

- **P5 (stride-32) branch removed**, backbone and head. Measured on the
  labelled tiles, the largest box is ~350 px against a 512 px tile (p99 is
  234 px). P4's regression range (reg_max 16 × stride 16 = 256 px per box
  side) already covers that; P5 exists for objects that fill the whole image
  and would sit idle while holding the widest, most parameter-heavy layers.
- **width 0.25 with channels capped at 192** (yolov8n: width 0.25, cap 1024).

For reference, yolov8n is 3.16 M parameters and yolo11n 2.62 M, so the cut is
roughly 8×. No published checkpoint fits this scaling, so the pretrain is made
to measure: the same architecture is first trained on COCO's vehicle classes
(allowed by rule 3), and those weights transfer into the fine-tune almost
whole (~97% of tensors; only the final class convolutions differ).

## Data

Training: the task's four clips (Pexels 8968356, 5382494, 8457857, 3405804)
plus one added clip (Pexels 15857456, ⟨TODO: why — e.g. 200–400 m band
coverage, rule 4 allows it⟩), all at native resolution. Frames are cut into
**512 px tiles** rather than downscaled: a distant vehicle is 10–20 px and
would vanish at 640-px full-frame inference. The same tiling runs at inference
(`eval/predict_video.py`, 128 px overlap, class-agnostic NMS on the merged
frame), so train and test see one object scale. Tiles are spread thin across
many frames instead of dense over few — the clips are 17–31 s, so frame
diversity is the scarce resource, not tile count (`labeling/build_dataset.py`).

Validation is a **whole held-out clip** (Pexels 14710575, labelled with the
same pipeline, never trained on): checkpoint and confidence threshold are
picked on footage the model has never seen, which a within-clip temporal split
cannot guarantee — neighbouring frames are near-duplicates. The eval clip
stays untouched by every one of these decisions (rule 1).
(`training/make_split.py` remains for a temporal split when only the four
clips are available.)

## Labels (rule 2)

No hand-drawn boxes from scratch; every box starts as model output and goes
through filters whose disagreements are cheap to review as crop contact
sheets. The stages, in `labeling/`:

1. **Propose** — YOLO-World with "…seen from above" prompts, tiled over full
   frames (`tiled_yoloworld.py`). Prompt phrasing roughly doubles recall over
   bare nouns; distractor prompts (lane paint, rooftops, kerbs) soak up the
   false positives. A custom intermediate detector and YOLOE run as second
   opinions (`pipeline_v1.py`, `agree_filter.py`): their errors barely
   overlap, so cross-model agreement beats either model's own confidence.
2. **Score** — every crop goes through a 2-class vehicle/non-vehicle
   classifier (`classify_crops.py`); one score decides acceptance regardless
   of which detector proposed the box.
3. **Recover & correct** — passes that chase specific, observed failure
   modes: YOLO-World top-ups for two-wheelers the other models are blind to
   (`yw_additions.py`, `yw_topup_v1.py`), re-admitting large vehicles an area
   cap had wrongly deleted (`bigbox_pass.py`), splitting road-sized boxes
   into the vehicles actually inside them (`split_big.py`), and MobileSAM
   box-refit with guards that keep a leaking mask from corrupting a good box
   (`refine_boxes.py`). Embedding similarity sweeps recurring junk (traffic
   cones, road markings) in one move (`separate_similar.py`).
4. **Export & eyeball** — labels rebuilt from whatever is currently accepted
   on disk (`export_labels.py`), rendered back onto the source frames as video
   (`render_valid.py`) for a final visual pass.

Correction tally (added / removed / fixed): **⟨TODO⟩ / ⟨TODO⟩ / ⟨TODO⟩**
boxes over ⟨TODO⟩ tiles.

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
are picked on the validation split only (rule 1).

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
# 1. predictions over the full confidence range (AP needs it)
python eval/predict_video.py --video <eval.mp4> --weights <best.pt> --conf 0.05
# 2. ground truth: label the eval clip with the same pipeline, then correct it;
#    store as csv: frame,x1,y1,x2,y2
# 3. banded metrics; --conf here is the shipping threshold from training data
python eval/eval_bands.py --gt gt.csv --pred eval_run/predictions.csv \
    --img-width 3840 --fps 30 --conf <shipping>
python eval/render_examples.py --video <eval.mp4> --gt gt.csv \
    --pred eval_run/predictions.csv --conf <shipping>
```

TP/FN land in the band of the GT box, FP in the band of the prediction's own
estimated distance. Time to first detection links GT boxes into tracks by IoU
across frames (no identity in the csv), then measures first-appearance to
first-match per track; the script also reports how many tracks were never
detected, which the median alone would hide.

Band coverage (rule 4): ⟨TODO — confirm the eval clip has vehicles in the
200–400 m band; if not, name the added clip here⟩.

## Repo layout

```
configs/     model yaml (the <500k detector)
labeling/    propose → score → correct → export, as described above
training/    temporal split + training with the parameter-budget assert
cloud/       RunPod launcher: same training, rented GPU, one command
eval/        tiled inference, distance model, banded metrics, example frames
examples/    eval frames, GT vs predictions
weights/     final checkpoint
```

Labeling and training scripts run **from a workspace directory** holding the
videos and datasets (paths resolve against the cwd), so the repo stays free of
multi-GB artefacts. `requirements.txt` + ffmpeg on PATH is the full setup.

## What next / decided against

- ⟨TODO: e.g. tracker-fed temporal smoothing at inference, quantisation for
  the target board, a P2 head experiment, more clips for band 200–400 m⟩
- Decided against DINO feature distillation for the backbone: no pretrained
  weights fit the custom scaling anyway, the gain at 400k parameters on a
  narrow domain does not pay for the compute, and the pseudo-label pipeline
  is already teacher→student distillation.

AI assistance: parts of the tooling and this README were written with an AI
assistant; every design decision and number was checked by hand. ⟨TODO: adjust
to taste.⟩
