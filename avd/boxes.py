"""Box geometry shared by the labelling filters and the tiled detector.

IoU asks "are these the same box", IoM "does one lie inside the other". Tile
seams leave fragments where IoU is tiny but IoM near 1, so the merge needs IoM
while de-duplication needs IoU.
"""

from .config import get

CONTAINMENT = float(get("inference.containment_merge", 0.7))
FRAME_NMS_IOU = float(get("inference.frame_nms_iou", 0.5))
MAX_OVERLAP = float(get("inference.max_overlap", 0.2))


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def iom(a, b):
    """Intersection over the smaller area: 1.0 when one box contains the other."""
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


def merge_overlapping(boxes, threshold=CONTAINMENT):
    """Fold each box into the first kept box that contains it, taking the union.

    Boxes are expected sorted by descending confidence, so the surviving box
    keeps the better score while growing to cover the whole vehicle.
    """
    kept = []
    for box in boxes:
        box = list(box)
        for i, other in enumerate(kept):
            if iom(box, other) > threshold:
                kept[i] = [min(box[0], other[0]), min(box[1], other[1]),
                           max(box[2], other[2]), max(box[3], other[3])]
                break
        else:
            kept.append(box)
    return kept


def nms(boxes, scores, iou_threshold):
    """Indices to keep, best score first. Vectorised, numpy in and numpy out.

    Both backends come through here, so they de-duplicate identically.
    """
    import numpy as np

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    areas = ((boxes[:, 2] - boxes[:, 0]).clip(min=0)
             * (boxes[:, 3] - boxes[:, 1]).clip(min=0))
    order = scores.argsort()[::-1]

    kept = []
    while order.size:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = (rb - lt).clip(min=0).prod(axis=1)
        union = areas[i] + areas[rest] - inter
        order = rest[np.divide(inter, union, out=np.zeros_like(inter),
                               where=union > 0) <= iou_threshold]
    return np.array(kept, dtype=np.int64)


def merge_detections(boxes, scores, frame_iou=None, containment=CONTAINMENT,
                     max_overlap=None):
    """Per-tile boxes in frame coordinates to one box per vehicle.

    NMS, then a containment merge (non-max merging, as in SAHI): a seam's
    part-box has too little IoU with the whole-vehicle box for plain NMS but
    lies almost inside it, so it is folded into the union of the two.
    """
    import numpy as np

    frame_iou = FRAME_NMS_IOU if frame_iou is None else frame_iou
    max_overlap = MAX_OVERLAP if max_overlap is None else max_overlap

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not len(boxes):
        return boxes, scores

    keep = nms(boxes, scores, frame_iou)
    boxes, scores = boxes[keep].copy(), scores[keep].copy()
    areas = ((boxes[:, 2] - boxes[:, 0]).clip(min=0)
             * (boxes[:, 3] - boxes[:, 1]).clip(min=0))

    kept = []
    for i in range(len(boxes)):
        for j in kept:
            lt = np.maximum(boxes[i, :2], boxes[j, :2])
            rb = np.minimum(boxes[i, 2:], boxes[j, 2:])
            inter = (rb - lt).clip(min=0).prod()
            if inter / max(min(areas[i], areas[j]), 1e-6) > containment:
                boxes[j, :2] = np.minimum(boxes[j, :2], boxes[i, :2])
                boxes[j, 2:] = np.maximum(boxes[j, 2:], boxes[i, 2:])
                areas[j] = ((boxes[j, 2] - boxes[j, 0])
                            * (boxes[j, 3] - boxes[j, 1]))
                break
        else:
            kept.append(i)

    boxes, scores = boxes[kept], scores[kept]
    keep = nms(boxes, scores, max_overlap)
    return boxes[keep], scores[keep]


def suppress_contained(boxes, threshold=CONTAINMENT):
    """Indices to keep: drop a box that lies mostly inside an earlier one."""
    kept = []
    for i, box in enumerate(boxes):
        if any(iom(box, boxes[j]) > threshold for j in kept):
            continue
        kept.append(i)
    return kept
