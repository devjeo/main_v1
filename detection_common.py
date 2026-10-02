"""
Shared detection utilities for Phase 4A (laptop server) and Phase 4B.4
(Pi local fallback) -- both paths wrap an ultralytics YOLO model the
same way and produce the same Detection shape, so that logic lives here
once instead of being duplicated between server_model.py and
pi_local_detection.py.
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

logger = logging.getLogger("detection_common")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list       # [x1, y1, x2, y2] in pixel coords
    position: str     # "left" / "center" / "right", relative to frame width
    track_id: Optional[int] = None  # persistent ID across frames when tracking is used; None if not tracked

    def to_dict(self):
        return asdict(self)


def dedup_key(det: Detection) -> tuple:
    """
    Identifies the same physical object across frames/detections, for
    deduplication -- prefers the tracker's persistent track_id (see
    run_yolo_detect below) when available, so the same object doesn't
    get counted twice just because it drifted across a left/center/right
    position boundary between frames. Falls back to (label, position)
    when track_id is None: the tracker hasn't confirmed an ID for this
    box yet, or these detections didn't come through tracking at all
    (e.g. CrosswalkModel, which never sets track_id).

    Shared between server_webrtc.py (buffers detections into a 2-second
    window, keyed by this) and the Pi side, so both machines agree on
    what "the same object" means.
    """
    if det.track_id is not None:
        return ("track", det.track_id)
    return ("label_pos", det.label, det.position)


def bbox_position(center_x: float, frame_width: int) -> str:
    """Buckets a bounding box's horizontal center into left/center/right thirds."""
    if center_x < frame_width * 0.33:
        return "left"
    elif center_x > frame_width * 0.66:
        return "right"
    return "center"


def run_yolo_detect(
    model, frame: np.ndarray, confidence_threshold: float, use_tracking: bool = True
) -> List[Detection]:
    """
    Runs a loaded ultralytics YOLO model on a single BGR frame (as
    returned by OpenCV/picamera2) and returns a list of Detection.

    Uses YOLO's built-in tracker (model.track(), ByteTrack by default)
    rather than plain model.predict() so each detection carries a
    persistent track_id across calls -- the same physical object keeps
    the same ID as it moves through frame (e.g. a person walking from
    center to left is still track 7, not treated as a brand-new
    detection). persist=True keeps the tracker's internal state on
    `model` between calls instead of resetting it every call, which is
    correct here since detect() is called repeatedly on a continuous
    stream of frames from the same camera, not on unrelated one-off
    images.

    use_tracking=False falls back to plain frame-by-frame detection
    (track_id always None) for callers with no real frame sequence to
    track across (e.g. a single test image).
    """
    if use_tracking:
        results = model.track(frame, conf=confidence_threshold, persist=True, verbose=False)
    else:
        results = model.predict(frame, conf=confidence_threshold, verbose=False)

    detections = []
    frame_width = frame.shape[1]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            center_x = (x1 + x2) / 2
            # box.id is None when the tracker hasn't confirmed an ID
            # for this box yet (e.g. brand new detection this frame).
            track_id = int(box.id[0]) if box.id is not None else None

            detections.append(Detection(
                label=label,
                confidence=round(conf, 3),
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                position=bbox_position(center_x, frame_width),
                track_id=track_id,
            ))

    return detections