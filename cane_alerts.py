"""
cane_alerts.py - emergency alerts + device logs from the Smart Cane to Supabase,
with an on-disk queue so nothing is lost when Wi-Fi is down.

  * emergency(kind, ...)  -> RPC device_emergency   (kind: sos | fall | hazard | fault)
  * log(level, message)   -> RPC device_log         (level: info | warning | error)
  * SOS button (optional) -> hold it ~1 s to fire emergency("sos") AND queue a
                             photo of what the camera sees (frame_provider)
  * photo                 -> sent to the cane-sos-photo Edge Function right after the
                             SOS is delivered; the server attaches it to that SOS log

How the queue works
  Every call is written to alerts_queue.jsonl FIRST, then a background thread
  sends it. If the network is down or Supabase answers 5xx, the item stays in
  the file and is retried (backoff 2s -> 60s), so it also survives a reboot.
  Emergencies are always sent before logs. If the queue passes MAX_QUEUE the
  oldest LOG is dropped first (an emergency is only dropped if nothing else is left).
  An alert that is delivered late gets "(sent N min late)" added to its text,
  because the server stamps created_at when it RECEIVES it.

The device secret is NOT written to the queue file; it is added at send time.
Reuses the config and RPC helper from cane_cloud.CaneCloud.

Wiring (default): push button between GPIO17 (physical pin 11) and GND
(physical pin 9). No resistor needed, the internal pull-up is used.
Needs gpiozero for the button (preinstalled on Raspberry Pi OS; otherwise
`sudo apt install python3-gpiozero`). Without it, everything else still works.
"""

import json
import os
import threading
import time

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE_FILE = os.path.join(_HERE, "alerts_queue.jsonl")
MAX_QUEUE = 200
PHOTO_DIR = os.path.join(_HERE, "sos_photos")
PHOTO_JPEG_QUALITY = 75
PHOTO_MAX_TRIES = 4        # tries when the server says "no SOS log yet"
SOS_PIN = 17               # BCM numbering
SOS_HOLD_S = 1.0           # hold this long so a bump in a bag doesn't fire it
SOS_LOCAL_COOLDOWN_S = 5.0 # ignore repeat presses inside this window
LATE_NOTE_AFTER_S = 60

KINDS = ("sos", "fall", "hazard", "fault")
LEVELS = ("info", "warning", "error")


class CaneAlerts:
    def __init__(self, cloud, audio=None, queue_path=QUEUE_FILE, sos_pin=SOS_PIN, frame_provider=None):
        """cloud: a cane_cloud.CaneCloud. audio: optional pi_audio.AudioOutput
        (used to speak SOS feedback). sos_pin=None disables the button.
        frame_provider: zero-arg callable returning the latest BGR camera frame
        (or None); used for the SOS photo. None = SOS without a photo."""
        self.cloud = cloud
        self.frame_provider = frame_provider
        self.audio = audio
        self.path = queue_path
        self.sos_pin = sos_pin

        self._lock = threading.Lock()
        self._items = self._load()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._button = None
        self._last_sos = 0.0

    # ------------------------------------------------------------------ queue
    def _load(self):
        items = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except ValueError:
                        pass  # skip a half-written line from a power cut
        return items

    def _save_locked(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for it in self._items:
                f.write(json.dumps(it) + "\n")
        os.replace(tmp, self.path)

    def _enqueue(self, rpc, payload):
        with self._lock:
            self._items.append({"rpc": rpc, "payload": payload, "ts": time.time()})
            while len(self._items) > MAX_QUEUE:
                idx = next((i for i, it in enumerate(self._items) if it["rpc"] == "device_log"),
                           next((i for i, it in enumerate(self._items) if it["rpc"] == "sos_photo"), 0))
                self._discard_photo(self._items.pop(idx))
            self._save_locked()
        self._wake.set()

    def _peek(self):
        with self._lock:
            if not self._items:
                return None
            for it in self._items:               # emergencies jump the line
                if it["rpc"] == "device_emergency":
                    return it
            return self._items[0]

    def _remove(self, item):
        with self._lock:
            try:
                self._items.remove(item)
            except ValueError:
                return
            self._save_locked()
        self._discard_photo(item)

    @staticmethod
    def _discard_photo(item):
        if item.get("rpc") == "sos_photo":
            try:
                os.remove(item["payload"].get("file") or "")
            except (OSError, TypeError):
                pass

    def pending(self):
        with self._lock:
            return len(self._items)

    # ------------------------------------------------------------- public API
    def emergency(self, kind, message=None, lat=None, lng=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self._enqueue("device_emergency", {
            "p_kind": kind, "p_message": message, "p_lat": lat, "p_lng": lng})
        print(f"[alert] queued emergency: {kind}")

    def log(self, level, message):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        self._enqueue("device_log", {"p_level": level, "p_message": message})

    def _queue_photo(self):
        """Grab the current camera frame NOW (at the moment of the press) and
        queue it. Never raises: SOS must work even if the photo can't."""
        if self.frame_provider is None:
            return
        try:
            frame = self.frame_provider()
            if frame is None:
                print("[alert] no camera frame available, SOS goes without a photo")
                return
            import cv2
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PHOTO_JPEG_QUALITY])
            if not ok:
                return
            os.makedirs(PHOTO_DIR, exist_ok=True)
            path = os.path.join(PHOTO_DIR, f"{int(time.time() * 1000)}.jpg")
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            self._enqueue("sos_photo", {"file": path})
            print("[alert] queued SOS photo")
        except Exception as e:
            print(f"[alert] could not queue SOS photo: {e}")

    # ----------------------------------------------------------------- sender
    def _send_photo(self, item):
        c = self.cloud
        if not c.configured:
            return "retry"
        try:
            with open(item["payload"]["file"], "rb") as f:
                jpeg = f.read()
        except (OSError, KeyError, TypeError):
            return "drop"                       # file is gone, nothing to send
        try:
            r = requests.post(
                f"{c.url}/functions/v1/cane-sos-photo",
                headers={"apikey": c.anon_key, "Authorization": f"Bearer {c.anon_key}"},
                data={"device_id": c.device_id, "secret": c.secret},
                files={"image": ("sos.jpg", jpeg, "image/jpeg")},
                timeout=20,
            )
        except requests.RequestException:
            return "retry"
        if r.ok:
            print("[alert] SOS photo uploaded")
            return "ok"
        if r.status_code == 409:                # SOS log not there (yet)
            item["tries"] = item.get("tries", 0) + 1
            if item["tries"] >= PHOTO_MAX_TRIES:
                print("[alert] no SOS log to attach the photo to (dropping photo)")
                return "drop"
            return "retry"
        if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
            print(f"[alert] photo refused (dropping): HTTP {r.status_code} {r.text[:200]}")
            return "drop"
        print(f"[alert] photo upload HTTP {r.status_code}, will retry")
        return "retry"

    def _send(self, item):
        """Returns 'ok', 'drop' (never going to work) or 'retry'."""
        if item["rpc"] == "sos_photo":
            return self._send_photo(item)
        c = self.cloud
        if not c.configured:
            return "retry"
        payload = dict(item["payload"])
        payload["p_device_id"] = c.device_id
        payload["p_secret"] = c.secret

        late_s = time.time() - item.get("ts", time.time())
        if LATE_NOTE_AFTER_S < late_s < 86400:     # ignore absurd values (clock not synced)
            note = f"(sent {int(late_s // 60)} min late)"
            msg = payload.get("p_message")
            payload["p_message"] = f"{msg} {note}" if msg else note

        try:
            r = c._rpc(item["rpc"], payload)
        except requests.RequestException:
            return "retry"
        if r.ok:
            return "ok"
        if r.status_code in (400, 422):
            print(f"[alert] server refused {item['rpc']} (dropping): {r.text[:200]}")
            return "drop"
        print(f"[alert] {item['rpc']} HTTP {r.status_code}, will retry")
        return "retry"

    def _run(self):
        backoff = 2
        while not self._stop.is_set():
            item = self._peek()
            if item is None:
                self._wake.wait(timeout=30)
                self._wake.clear()
                continue
            outcome = self._send(item)
            if outcome in ("ok", "drop"):
                self._remove(item)
                backoff = 2
                if outcome == "ok" and item["payload"].get("p_kind") == "sos":
                    print("[alert] SOS delivered")
                    self._say("Alert delivered.")
            else:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    # ------------------------------------------------------------ SOS button
    def _say(self, text):
        if self.audio is None:
            return
        try:
            from pi_audio import Priority
            self.audio.speak(text, Priority.EMERGENCY)
        except Exception as e:
            print(f"[alert] could not speak: {e}")

    def _on_sos(self):
        now = time.monotonic()
        if now - self._last_sos < SOS_LOCAL_COOLDOWN_S:
            return
        self._last_sos = now
        self.emergency("sos", "SOS button pressed")
        self._queue_photo()          # queued AFTER the SOS, so the SOS always goes first
        self._say("S O S. Sending alert.")

    def _start_button(self):
        if self.sos_pin is None:
            return
        try:
            from gpiozero import Button
            self._button = Button(self.sos_pin, pull_up=True,
                                  bounce_time=0.05, hold_time=SOS_HOLD_S)
            self._button.when_held = self._on_sos
            print(f"[alert] SOS button ready on GPIO{self.sos_pin} (hold {SOS_HOLD_S:g}s)")
        except Exception as e:
            print(f"[alert] SOS button disabled: {e}")

    # -------------------------------------------------------------- lifecycle
    def start(self):
        self._start_button()
        self._thread = threading.Thread(target=self._run, name="cane-alerts", daemon=True)
        self._thread.start()
        if self._items:
            print(f"[alert] {len(self._items)} queued item(s) from last run, sending")

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._button is not None:
            try:
                self._button.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)


if __name__ == "__main__":
    # Real test: python cane_alerts.py   (needs cane.env; hold the button, or Ctrl+C)
    from cane_cloud import CaneCloud
    cloud = CaneCloud()
    alerts = CaneAlerts(cloud)
    alerts.start()
    alerts.log("info", "cane_alerts self-test")
    print("Running. Hold the SOS button to test, Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        alerts.stop()