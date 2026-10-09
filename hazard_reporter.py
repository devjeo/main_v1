"""
hazard_reporter.py -- uploads a photo to Supabase when a DANGEROUS object is close.

Put next to ai_stream.py. Needs: requests, opencv (already used).

Flow:  detection loop -> consider(frame, detections) -> small queue -> worker
       thread -> POST to the `cane-hazard` Edge Function (multipart JPEG).
The server attaches the phone's location and writes the log row.

Anti-spam (a car in view for a minute must not make 60 uploads):
  * only labels in DANGEROUS_LABELS
  * box must cover >= MIN_AREA_RATIO of the frame (i.e. it is actually close)
  * at most one upload per label every COOLDOWN_S
  * a photo that could not be sent within MAX_AGE_S is dropped (stale)
Never blocks or crashes the detection loop.
"""
import queue
import threading
import time

import cv2
import requests

DANGEROUS_LABELS = {"car", "motorcycle", "truck", "bus", "train", "bicycle"}
MIN_CONFIDENCE = 0.6
MIN_AREA_RATIO = 0.08      # box area / frame area; ~0.08 is "a few metres away" -- tune on the cane
COOLDOWN_S = 45
MAX_AGE_S = 120
JPEG_QUALITY = 70
UPLOAD_TIMEOUT_S = 15


class HazardReporter:
    def __init__(self, cloud, labels=DANGEROUS_LABELS, min_conf=MIN_CONFIDENCE,
                 min_area=MIN_AREA_RATIO, cooldown_s=COOLDOWN_S, annotate=None):
        """cloud: CaneCloud. annotate: optional fn(frame, detections) -> frame with boxes drawn."""
        self.cloud = cloud
        self.labels = set(labels)
        self.min_conf = min_conf
        self.min_area = min_area
        self.cooldown_s = cooldown_s
        self.annotate = annotate
        self._last_sent = {}
        self._q = queue.Queue(maxsize=3)
        self._stop = threading.Event()
        self._thread = None

    # ----------------------------------------------------------- detection side
    def consider(self, frame, detections):
        """Call once per frame. Cheap unless it actually decides to report."""
        if not self.cloud.configured:
            return
        h, w = frame.shape[:2]
        best, best_area = None, 0.0
        for d in detections:
            if d.label not in self.labels or d.confidence < self.min_conf:
                continue
            x1, y1, x2, y2 = d.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(w * h)
            if area >= self.min_area and area > best_area:
                best, best_area = d, area
        if best is None:
            return
        now = time.monotonic()
        if now - self._last_sent.get(best.label, float("-inf")) < self.cooldown_s:
            return
        self._last_sent[best.label] = now

        img = self.annotate(frame, detections) if self.annotate else frame
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        item = {"jpeg": buf.tobytes(), "label": best.label, "confidence": best.confidence,
                "position": best.position, "ts": time.monotonic()}
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()      # drop the oldest, keep the newest
            except queue.Empty:
                pass
            self._q.put_nowait(item)
        print(f"[hazard] queued {best.label} ({best.confidence:.2f}, {best_area:.0%} of frame)")

    # ------------------------------------------------------------- sender side
    def _post(self, item):
        c = self.cloud
        r = requests.post(
            f"{c.url}/functions/v1/cane-hazard",
            headers={"apikey": c.anon_key, "Authorization": f"Bearer {c.anon_key}"},
            data={"device_id": c.device_id, "secret": c.secret, "label": item["label"],
                  "confidence": f"{item['confidence']:.3f}", "position": item["position"]},
            files={"image": ("hazard.jpg", item["jpeg"], "image/jpeg")},
            timeout=UPLOAD_TIMEOUT_S,
        )
        return r

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            backoff = 2
            while not self._stop.is_set() and time.monotonic() - item["ts"] < MAX_AGE_S:
                try:
                    r = self._post(item)
                except requests.RequestException as e:
                    print(f"[hazard] upload failed, retrying: {e}")
                else:
                    if r.ok:
                        print(f"[hazard] uploaded {item['label']}: {r.text[:120]}")
                        break
                    if 400 <= r.status_code < 500 and r.status_code != 429:
                        print(f"[hazard] server refused (dropping): {r.status_code} {r.text[:120]}")
                        break
                    print(f"[hazard] HTTP {r.status_code}, retrying")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 20)
            else:
                print(f"[hazard] dropped stale {item['label']} photo")

    def start(self):
        self._thread = threading.Thread(target=self._run, name="hazard-upload", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


if __name__ == "__main__":
    # End-to-end test: python hazard_reporter.py [label]   (needs cane.env). Creates a REAL log row.
    import sys
    import numpy as np
    from cane_cloud import CaneCloud
    from detection_common import Detection

    cloud = CaneCloud()
    rep = HazardReporter(cloud, min_area=0.0)
    rep.start()
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    cv2.putText(frame, "TEST", (240, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    rep.consider(frame, [Detection(sys.argv[1] if len(sys.argv) > 1 else "car", 0.9,
                                   [100, 100, 500, 400], "center", 1)])
    time.sleep(8)
    rep.stop()
