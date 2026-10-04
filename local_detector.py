"""
Local Detector -- runs YOLO (your NCNN export) directly on the Raspberry Pi
and returns Detection objects that carry a persistent tracker ID.

This replaces the old client/server detection path (pi_webrtc_client /
pi_detection_client + server_webrtc): nothing leaves the Pi.

Why model.track() and not model.predict():
    track() runs a tracker (ByteTrack by default here) on top of YOLO, so
    the same physical object keeps the same ID from frame to frame. That
    ID is what object_announcer.py uses to avoid saying the same object
    twice -- and to still say a second phone when it appears.

Requires: ultralytics. ByteTrack also needs the `lap` package; ultralytics
tries to auto-install it on the first track() call. If that fails on the
Pi (no internet / no wheel), `pip install lapx` is the usual fix.
"""

import logging
from typing import List

import numpy as np

from detection_common import Detection, bbox_position

logger = logging.getLogger("local_detector")


class LocalTracker:
    def __init__(
        self,
        model_path: str = "models/yolo26n_ncnn_model",
        confidence: float = 0.5,
        imgsz: int = 320,
        tracker: str = "bytetrack.yaml",
        labels=None,
    ):
        """
        labels: optional collection of class NAMES (e.g. {"person", "chair"}).
        When given, ONLY these objects are ever returned -- so they are the
        only ones announced AND the only ones drawn in the browser view.
        None means no filtering (every class the model knows).
        """
        from ultralytics import YOLO  # imported here so the other modules stay importable without it

        self.confidence = confidence
        self.imgsz = imgsz
        self.tracker = tracker
        logger.info("Loading model %s (imgsz=%d, conf=%.2f, tracker=%s)",
                    model_path, imgsz, confidence, tracker)
        self.model = YOLO(model_path, task="detect")

        # Label filter. Matching is by NAME against the model's own class
        # list, so it can't drift out of sync with the model's numeric IDs.
        self.labels = None
        self.class_ids = None
        if labels is not None:
            self.labels = set(labels)
            names = self.model.names  # {class_id: name}
            self.class_ids = sorted(i for i, n in names.items() if n in self.labels)
            unknown = sorted(self.labels - set(names.values()))
            if unknown:
                logger.warning("These labels are NOT in this model and will never match: %s", unknown)
            if not self.class_ids:
                logger.warning("None of the requested labels exist in this model -- nothing will be reported!")
            logger.info("Label filter on: reporting %d of %d classes",
                        len(self.class_ids), len(names))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        One BGR frame in, list of Detection out. track_id is None for a box
        the tracker hasn't confirmed yet (normal for a brand-new object's
        first frame). persist=True keeps tracker state between calls, which
        is what makes IDs stable across a continuous camera stream.
        """
        results = self.model.track(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker,
            classes=self.class_ids or None,   # skip unwanted classes inside YOLO (None = all)
            verbose=False,
        )

        detections: List[Detection] = []
        frame_width = frame.shape[1]
        for result in results:
            for box in result.boxes:
                label = self.model.names[int(box.cls[0])]
                if self.labels is not None and label not in self.labels:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    label=label,
                    confidence=round(float(box.conf[0]), 3),
                    bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    position=bbox_position((x1 + x2) / 2, frame_width),
                    track_id=int(box.id[0]) if box.id is not None else None,
                ))
        return detections


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "models/yolo26n_ncnn_model"
    print(f"Testing LocalTracker with {path}\n")
    tracker = LocalTracker(model_path=path)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    print(f"OK -- model loaded; {len(tracker.detect(blank))} detections on a blank frame (expected 0).")