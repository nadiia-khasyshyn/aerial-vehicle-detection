"""Distance-banded metrics on a clip: ground-truth csv vs predictions csv.

Estimates a distance for every box (`distance.py` holds the model) and reports
each metric separately for 0-200 m and 200-400 m. Feed it a prediction csv
written at a low threshold, because AP needs the full confidence range; the
point metrics are then computed at --conf.

Matching is per frame, greedy by confidence, IoU >= --iou, one GT per
prediction. A TP or FN lands in the band of its GT box, an FP in the band of
its own estimated distance, and boxes outside every band are excluded.

Time to first detection needs identity the csv does not carry, so GT boxes are
linked into tracks by IoU across consecutive annotated frames; tracks shorter
than --min-track frames are skipped.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from distance import (BAND_NAMES, DEFAULT_HFOV_DEG, DEFAULT_VEHICLE_LEN_M,
                      band_index, box_distance_m)

N_BANDS = len(BAND_NAMES)


def read_boxes(path, with_conf):
    """frame -> list of [x1, y1, x2, y2] (+conf), extra csv columns ignored."""
    per_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
        if with_conf:
            box.append(float(r["conf"]))
        per_frame[int(r["frame"])].append(box)
    return per_frame


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def match_frame(gts, preds, thr):
    """Greedy by confidence. Returns (gt_index or None) per pred, matched gt set."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    taken = set()
    assign = [None] * len(preds)
    for i in order:
        best, best_iou = None, thr
        for j, g in enumerate(gts):
            if j in taken:
                continue
            v = iou(preds[i], g)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            taken.add(best)
            assign[i] = best
    return assign, taken


def build_tracks(gt, frames, track_iou):
    """Greedy IoU linking over consecutive annotated frames -> track id maps."""
    track_of = {}  # (frame, gt_index) -> track id
    next_id = 0
    prev = []      # (gt_index_in_prev_frame, track_id)
    prev_frame = None
    for f in frames:
        cur = []
        used = set()
        for j, g in enumerate(gt.get(f, [])):
            tid = None
            if prev_frame is not None:
                best, best_iou = None, track_iou
                for pj, ptid in prev:
                    if ptid in used:
                        continue
                    v = iou(g, gt[prev_frame][pj])
                    if v >= best_iou:
                        best, best_iou = ptid, v
                if best is not None:
                    tid = best
                    used.add(best)
            if tid is None:
                tid = next_id
                next_id += 1
            track_of[(f, j)] = tid
            cur.append((j, tid))
        prev, prev_frame = cur, f
    return track_of


def average_precision(records, n_gt):
    """records: (conf, is_tp) for one band; VOC-style continuous AP@IoU."""
    if n_gt == 0 or not records:
        return float("nan")
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # monotonic precision envelope, then area under PR
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="csv: frame,x1,y1,x2,y2")
    ap.add_argument("--pred", required=True,
                    help="csv from predict_video.py run at a low threshold")
    ap.add_argument("--img-width", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True,
                    help="fps of the ANNOTATED frames (video fps / frame stride)")
    ap.add_argument("--conf", type=float, required=True,
                    help="shipping threshold, decided on training data")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG)
    ap.add_argument("--vehicle-len", type=float, default=DEFAULT_VEHICLE_LEN_M)
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--frames", choices=["gt", "union"], default="gt",
                    help="'gt' evaluates only annotated frames; 'union' also "
                         "counts predictions on frames with no ground truth, "
                         "which is only right when every frame is annotated")
    ap.add_argument("--out", default=None, help="write metrics as json here")
    args = ap.parse_args()

    gt = read_boxes(args.gt, with_conf=False)
    pred_all = read_boxes(args.pred, with_conf=True)
    frames = (sorted(gt) if args.frames == "gt"
              else sorted(set(gt) | set(pred_all)))
    if not frames:
        raise SystemExit("no frames in either csv")
    extra = len(set(pred_all) - set(gt))
    if extra and args.frames == "gt":
        print(f"note: ignoring predictions on {extra} unannotated frames "
              f"(--frames union would count them as false alarms)")

    def dist(box):
        return box_distance_m(*box[:4], args.img_width, args.hfov,
                              args.vehicle_len)

    def band(box):
        return band_index(dist(box))

    # --- point metrics at the shipping threshold --------------------------
    tp = [0] * N_BANDS
    fn = [0] * N_BANDS
    fp = [0] * N_BANDS
    out_of_band = {"gt": 0, "pred": 0}
    matched_gt_frames = set()   # (frame, gt_index) matched at shipping conf

    for f in frames:
        gts = gt.get(f, [])
        preds = [p for p in pred_all.get(f, []) if p[4] >= args.conf]
        assign, taken = match_frame(gts, preds, args.iou)
        for j, g in enumerate(gts):
            b = band(g)
            if b is None:
                out_of_band["gt"] += 1
                continue
            if j in taken:
                tp[b] += 1
                matched_gt_frames.add((f, j))
            else:
                fn[b] += 1
        for i, p in enumerate(preds):
            if assign[i] is None:
                b = band(p)
                if b is None:
                    out_of_band["pred"] += 1
                else:
                    fp[b] += 1

    minutes = len(frames) / args.fps / 60.0

    # --- AP@IoU over the full confidence range ----------------------------
    ap_records = [[] for _ in range(N_BANDS)]
    n_gt_band = [0] * N_BANDS
    for f in frames:
        gts = gt.get(f, [])
        for g in gts:
            b = band(g)
            if b is not None:
                n_gt_band[b] += 1
        preds = pred_all.get(f, [])
        assign, _ = match_frame(gts, preds, args.iou)
        for i, p in enumerate(preds):
            if assign[i] is not None:
                b = band(gts[assign[i]])
                if b is not None:
                    ap_records[b].append((p[4], True))
            else:
                b = band(p)
                if b is not None:
                    ap_records[b].append((p[4], False))
    ap50 = [average_precision(ap_records[b], n_gt_band[b])
            for b in range(N_BANDS)]
    # The task's bonus metric is a single mAP@0.5 "over both bands".
    ap50_both = average_precision(ap_records[0] + ap_records[1],
                                  n_gt_band[0] + n_gt_band[1])

    # --- time to first detection over GT tracks ---------------------------
    gt_frames = sorted(set(gt))
    track_of = build_tracks(gt, gt_frames, args.track_iou)
    track_frames = collections.defaultdict(list)   # tid -> annotated frames
    track_hits = collections.defaultdict(list)     # tid -> frames matched
    track_band = {}
    for (f, j), tid in sorted(track_of.items()):
        track_frames[tid].append(f)
        if tid not in track_band:
            track_band[tid] = band(gt[f][j])
        if (f, j) in matched_gt_frames:
            track_hits[tid].append(f)

    ttfd = [[] for _ in range(N_BANDS)]
    never = [0] * N_BANDS
    n_tracks = [0] * N_BANDS
    for tid, fs in track_frames.items():
        b = track_band[tid]
        if b is None or len(fs) < args.min_track:
            continue
        n_tracks[b] += 1
        hits = track_hits.get(tid)
        if not hits:
            never[b] += 1
            continue
        ttfd[b].append((min(hits) - min(fs)) / args.fps)

    # --- report ------------------------------------------------------------
    def fmt(v, spec=".2f"):
        return format(v, spec) if v == v else "n/a"   # NaN-safe

    rows = []
    for b in range(N_BANDS):
        det_rate = tp[b] / (tp[b] + fn[b]) if tp[b] + fn[b] else float("nan")
        prec = tp[b] / (tp[b] + fp[b]) if tp[b] + fp[b] else float("nan")
        fa_min = fp[b] / minutes if minutes else float("nan")
        # The task sheet's formula is FP x 60 / N_frames — false alarms per 60
        # *frames*, which only equals "per minute" at 1 fps. Report both.
        fa_task = fp[b] * 60.0 / len(frames) if frames else float("nan")
        med = float(np.median(ttfd[b])) if ttfd[b] else float("nan")
        rows.append({
            "band": BAND_NAMES[b],
            "detection_rate": det_rate, "precision": prec,
            "false_alarms_per_min": fa_min, "fa_task_formula": fa_task,
            "ttfd_median_s": med,
            "mAP50": ap50[b],
            "tp": tp[b], "fn": fn[b], "fp": fp[b], "n_gt": n_gt_band[b],
            "tracks": n_tracks[b], "tracks_never_detected": never[b],
        })

    print(f"frames evaluated: {len(frames)}  ({minutes:.2f} min at {args.fps} fps)")
    print(f"shipping conf: {args.conf}, IoU: {args.iou}, "
          f"hfov {args.hfov} deg, vehicle length {args.vehicle_len} m")
    if out_of_band["gt"] or out_of_band["pred"]:
        print(f"outside 0-400 m (excluded): {out_of_band['gt']} GT, "
              f"{out_of_band['pred']} unmatched predictions")
    print()
    hdr = ["Metric"] + list(BAND_NAMES)
    table = [
        ("Detection rate TP/(TP+FN)", [fmt(r["detection_rate"]) for r in rows]),
        ("Precision TP/(TP+FP)", [fmt(r["precision"]) for r in rows]),
        ("False alarms / min of footage", [fmt(r["false_alarms_per_min"], ".1f") for r in rows]),
        ("False alarms FPx60/N_frames (task formula)", [fmt(r["fa_task_formula"], ".1f") for r in rows]),
        ("Time to first detection (median s)", [fmt(r["ttfd_median_s"]) for r in rows]),
        ("mAP@0.5", [fmt(r["mAP50"]) for r in rows]),
    ]
    w0 = max(len(t[0]) for t in table + [(hdr[0], None)])
    print(f"| {hdr[0]:<{w0}} | " + " | ".join(f"{h:>8}" for h in hdr[1:]) + " |")
    print(f"|{'-' * (w0 + 2)}|" + "|".join("-" * 10 for _ in BAND_NAMES) + "|")
    for name, vals in table:
        print(f"| {name:<{w0}} | " + " | ".join(f"{v:>8}" for v in vals) + " |")
    print()
    print(f"mAP@0.5 over both bands (task bonus): {fmt(ap50_both)}")
    for r in rows:
        print(f"{r['band']}: GT {r['n_gt']}, TP {r['tp']}, FN {r['fn']}, "
              f"FP {r['fp']}; tracks {r['tracks']} "
              f"({r['tracks_never_detected']} never detected)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "bands": rows,
             "mAP50_both_bands": ap50_both}, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
