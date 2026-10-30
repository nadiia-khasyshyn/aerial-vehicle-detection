"""Box geometry shared by the labelling filters and the tiled detector.

Two overlap measures are used throughout, and they answer different questions:
IoU asks "are these the same box", IoM asks "does one lie inside the other".
Tile seams produce fragments where IoU is tiny but IoM is close to 1, so the
merge step needs IoM while plain de-duplication needs IoU.
"""

from .config import get

CONTAINMENT = float(get("inference.containment_merge", 0.7))


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


def suppress_contained(boxes, threshold=CONTAINMENT):
    """Indices to keep: drop a box that lies mostly inside an earlier one."""
    kept = []
    for i, box in enumerate(boxes):
        if any(iom(box, boxes[j]) > threshold for j in kept):
            continue
        kept.append(i)
    return kept
