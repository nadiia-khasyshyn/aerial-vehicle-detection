"""The detector behind onnxruntime, with no torch and no ultralytics.

This is what the container runs. The graph exported from the checkpoint ends
at the raw head output — (batch, 4 + classes, anchors), boxes as xywh in input
pixels — so the decode and the per-tile NMS that ultralytics would normally do
are here instead, in numpy. `training/export_onnx.py` checks the two backends
produce the same boxes.

Preprocessing has to match what training did, or the parity check is the first
thing to notice: BGR to RGB, CHW, divided by 255, no mean/std normalisation.
"""

import numpy as np

from .boxes import nms
from .config import get

TILE = int(get("tiling.tile", 512))
NMS_IOU = float(get("inference.tile_nms_iou", 0.7))
PAD_VALUE = 114      # the grey ultralytics letterboxes with


class OnnxDetector:
    """Tile in, boxes out. Batches are the unit of work, one tile is a batch of one."""

    def __init__(self, model_path, providers=None, nms_iou=NMS_IOU, max_det=300,
                 threads=0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=providers or ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # A dynamic axis comes back as a string; fall back to the configured tile.
        self.tile = shape[2] if isinstance(shape[2], int) else TILE
        self.nms_iou = nms_iou
        self.max_det = max_det

    def _preprocess(self, tiles):
        batch = np.empty((len(tiles), 3, self.tile, self.tile), dtype=np.float32)
        for i, img in enumerate(tiles):
            h, w = img.shape[:2]
            if (h, w) != (self.tile, self.tile):
                padded = np.full((self.tile, self.tile, 3), PAD_VALUE, np.uint8)
                padded[:h, :w] = img[:self.tile, :self.tile]
                img = padded
            batch[i] = img[:, :, ::-1].transpose(2, 0, 1)
        return batch / 255.0

    def _decode(self, raw, conf):
        """One tile's raw head output to (boxes xyxy, scores), NMS applied."""
        scores = raw[4:].max(axis=0) if raw.shape[0] > 5 else raw[4]
        keep = scores >= conf
        if not keep.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        xywh, scores = raw[:4, keep].T, scores[keep]

        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        order = nms(boxes, scores, self.nms_iou)[:self.max_det]
        return boxes[order], scores[order]

    def detect_tiles(self, tiles, conf):
        """Boxes and scores per tile, in that tile's own pixel coordinates."""
        if not tiles:
            return []
        raw = self.session.run(None, {self.input_name: self._preprocess(tiles)})[0]
        return [self._decode(raw[i], conf) for i in range(len(tiles))]

    def detect_tile(self, tile, conf):
        return self.detect_tiles([tile], conf)[0]
