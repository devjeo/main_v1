"""
ai_stream.py -- THE MAIN PROGRAM. Run this one file and everything starts:

    python ai_stream.py

What it does, all on the Raspberry Pi (no laptop, no server):

  1. Camera + YOLO (NCNN) + tracker: every detected object gets an ID.
  2. Speech: objects are gathered for 3 seconds, then the NEW ones are
     spoken through Piper. The same object ID is never repeated within
     10 seconds (a second phone is a new ID, so it IS announced).
     Rules live in object_announcer.py.
  3. Browser view: open http://<pi-ip>:5000/ for the live annotated video
     (boxes + labels + IDs). Detection and speech run all the time in the
     background -- they do NOT depend on anyone having the page open.
  4. Online status: a heartbeat tells Supabase the cane is alive, and
     marks it offline on a normal shutdown (cane_cloud.py). Skipped with a
     note if cane.env isn't set up.

If the voice/audio can't start, the video + detection still run (speech is
simply skipped). If cane.env is missing, only the heartbeat is skipped.
"""

import argparse
import logging
import sys
import threading
import time

import cv2
from flask import Flask, Response

from cane_cloud import CaneCloud
from local_detector import LocalTracker
from narration_common import template_narrate
from object_announcer import ObjectAnnouncer
from pi_audio import AudioError, AudioOutput, Priority

# ---------------------------------------------------------------------
# CONFIG -- edit for your setup
# ---------------------------------------------------------------------
MODEL_PATH = "models/yolo26n_ncnn_model"   # your NCNN export
PIPER_DIR = "./piper"                      # folder containing voices/
TTS_CACHE_DIR = "./tts_cache"              # spoken phrases are cached here across restarts
HOST = "0.0.0.0"
PORT = 5000
# ---------------------------------------------------------------------

logger = logging.getLogger("ai_stream")
app = Flask(__name__)


class StreamState:
    """Latest annotated JPEG for the browser, shared between the detection
    thread (writer) and the Flask video route (readers)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = b""
        self.seq = 0        # bumps every new frame so viewers don't resend an old one
        self.viewers = 0    # JPEG encoding only happens while someone is watching


state = StreamState()


# ----------------------------------------------------------------------
# camera (your original setup, plus a 1-frame buffer so frames aren't stale)
# ----------------------------------------------------------------------
def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def draw_boxes(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 136), 2)
        tag = f"{d.label} #{d.track_id}" if d.track_id is not None else d.label
        cv2.putText(out, tag, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
    return out


# ----------------------------------------------------------------------
# background loop: capture -> detect -> gather -> speak (+ feed the browser)
# ----------------------------------------------------------------------
def detection_loop(stop, cap, detector, announcer, audio, max_fps, cooldown_s):
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    read_failures = 0

    while not stop.is_set():
        loop_start = time.monotonic()
        try:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures == 30:
                    logger.warning("Camera isn't returning frames -- check the connection.")
                announcer.poll()  # keep the window clock honest even without frames
                time.sleep(0.1)
                continue
            read_failures = 0

            detections = detector.detect(frame)
            announcer.add(detections)

            result = announcer.poll()
            if result is not None:
                if result.announce:
                    text = template_narrate(result.announce)
                    print(f"[say]  {text}")
                    if audio is not None:
                        audio.speak(text, Priority.NARRATION, objects=result.announce)
                if result.suppressed:
                    held = ", ".join(f"{d.label}#{d.track_id}" for d in result.suppressed)
                    print(f"[held] announced < {cooldown_s:g}s ago: {held}")

            with state.lock:
                has_viewers = state.viewers > 0
            if has_viewers:
                ok_jpg, buf = cv2.imencode(".jpg", draw_boxes(frame, detections),
                                           [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_jpg:
                    with state.lock:
                        state.jpeg = buf.tobytes()
                        state.seq += 1
        except Exception:
            logger.exception("Detection loop error (continuing)")
            time.sleep(0.5)

        spare = min_interval - (time.monotonic() - loop_start)
        if spare > 0:
            time.sleep(spare)


# ----------------------------------------------------------------------
# web page (same look as before)
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Cane AI Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
            }

            h1 {
                margin-top: 20px;
            }

            .status {
                color: #00ff88;
                margin-bottom: 10px;
            }

            img {
                width: 95%;
                max-width: 900px;
                border-radius: 10px;
                margin-top: 10px;
            }
        </style>
    </head>

    <body>
        <h1>Smart Cane AI Detection</h1>
        <div class="status">LIVE</div>
        <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    def generate():
        with state.lock:
            state.viewers += 1
        last_seq = -1
        try:
            while True:
                with state.lock:
                    jpeg, seq = state.jpeg, state.seq
                if jpeg and seq != last_seq:
                    last_seq = seq
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(0.05)
        finally:
            with state.lock:
                state.viewers -= 1

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smart Cane: detection + speech + live view")
    p.add_argument("--model", default=MODEL_PATH, help="path to the YOLO / NCNN model")
    p.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=320, help="inference image size")
    p.add_argument("--max-fps", type=float, default=10, help="cap on detection frames/sec (leaves CPU for speech)")
    p.add_argument("--window", type=float, default=3.0, help="gather window in seconds")
    p.add_argument("--cooldown", type=float, default=10.0, help="seconds before the SAME object ID may be spoken again")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="bytetrack.yaml (lighter) or botsort.yaml (better with camera motion, more CPU)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main(args):
    # Online-status heartbeat (does nothing, with a note, if cane.env is missing).
    cloud = CaneCloud()
    cloud.install_shutdown_hooks()  # must happen on the main thread
    cloud.start()

    cap = open_camera()
    if not cap.isOpened():
        sys.exit("Could not open the camera (/dev/video0). Is it plugged in?")

    detector = LocalTracker(model_path=args.model, confidence=args.conf,
                            imgsz=args.imgsz, tracker=args.tracker)
    announcer = ObjectAnnouncer(window_s=args.window, cooldown_s=args.cooldown)

    audio = None
    try:
        audio = AudioOutput(piper_dir=PIPER_DIR, cache_dir=TTS_CACHE_DIR)
        audio.start()
    except AudioError as e:
        print(f"[warn] Speech disabled: {e}\n       (video + detection will still run)")

    stop = threading.Event()
    worker = threading.Thread(
        target=detection_loop,
        args=(stop, cap, detector, announcer, audio, args.max_fps, args.cooldown),
        name="detection", daemon=True,
    )
    worker.start()

    print(f"\nRunning. Live view: http://<pi-ip>:{PORT}/   (Ctrl+C to stop)")
    print(f"Gathering {args.window:g}s windows; same object ID won't repeat within {args.cooldown:g}s.\n")

    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    finally:
        stop.set()
        worker.join(timeout=3)
        cap.release()
        if audio is not None:
            audio.stop()
        cloud.shutdown()


if __name__ == "__main__":
    cli = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if cli.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main(cli)