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
    ):
        from ultralytics import YOLO  # imported here so the other modules stay importable without it

        self.confidence = confidence
        self.imgsz = imgsz
        self.tracker = tracker
        logger.info("Loading model %s (imgsz=%d, conf=%.2f, tracker=%s)",
                    model_path, imgsz, confidence, tracker)
        self.model = YOLO(model_path, task="detect")

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
            verbose=False,
        )

        detections: List[Detection] = []
        frame_width = frame.shape[1]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    label=self.model.names[int(box.cls[0])],
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