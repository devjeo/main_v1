r"""
Phase 3.3-3.4 -- Audio Output & TTS (Piper edition)

Uses Piper (https://github.com/rhasspy/piper) as the TTS engine -- offline,
good voice quality, cross-platform. The voice model is loaded ONCE, at
startup, via Piper's Python API (`pip install piper-tts`) and kept warm
in memory for the lifetime of the process. Each speak() call reuses that
already-loaded model to synthesize straight to an in-memory WAV buffer,
which is then handed to the platform's native player -- so the same code
runs unchanged on your Windows dev machine now and on the Pi (Linux)
later.

WHY "WARM"?
  The previous version of this module shelled out to the `piper` CLI
  binary via subprocess.run() for every single utterance. That meant
  every speak() call paid the FULL cost of loading the ONNX model off
  disk and initializing onnxruntime from scratch -- on a Raspberry Pi's
  CPU, that load/init cost can rival or exceed the actual synthesis
  time. Loading the model once here and reusing it removes that
  per-utterance reload entirely; speak() now only pays for inference.

REQUIRES:
  pip install piper-tts

  plus a voice model:
  - a `piper_dir` pointing at a folder laid out like:
        piper/
          voices/
            <name>.onnx
            <name>.onnx.json
    (this matches the layout used by rhasspy/piper release bundles --
    if you've used Piper on a previous project, you likely already have
    a folder like this and don't need to download anything new)
  - or `model_path` pointing directly at a `<name>.onnx` file (with a
    matching `<name>.onnx.json` next to it), or the PIPER_VOICE_MODEL
    environment variable set to that path.

  If nothing is downloaded yet, get a starter voice (English, US, medium)
  from https://huggingface.co/rhasspy/piper-voices -- you need both the
  .onnx file and its matching .onnx.json config file.
"""

import glob
import hashlib
import io
import json
import logging
import os
import platform
import queue
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("pi_audio")

SYSTEM = platform.system()  # "Windows", "Darwin", or "Linux"


class Priority(IntEnum):
    NARRATION = 0   # scene descriptions -- lowest priority, interruptible
    INFO = 1        # e.g. "server unavailable, switching to local mode"
    ALERT = 2       # zone obstacle warnings (down/forward/up)
    EMERGENCY = 3   # emergency button confirmation -- highest, always heard


@dataclass(order=True)
class _QueuedItem:
    sort_key: int
    timestamp: float = field(compare=False)
    text: str = field(compare=False)
    priority: int = field(compare=False)
    pattern_key: Optional[Tuple] = field(compare=False, default=None)


class AudioError(Exception):
    pass


# A pattern key is an order-independent, count-aware fingerprint of "what
# objects, at what positions, were in this batch" -- e.g. two chairs at
# center and one person on the left always produce the same key no matter
# what order they were detected in. It's built from anything with
# .label/.position attributes (Detection instances), (label, position)
# tuples, or {"label":..., "position":...} dicts, so pi_audio.py doesn't
# need to import detection_common.py just to accept a batch.
PatternKey = Tuple[Tuple[str, str, int], ...]


def make_pattern_key(objects: Iterable) -> PatternKey:
    counts: Dict[Tuple[str, str], int] = {}
    for obj in objects:
        if hasattr(obj, "label") and hasattr(obj, "position"):
            label, position = obj.label, obj.position
        elif isinstance(obj, dict):
            label, position = obj["label"], obj["position"]
        else:
            label, position = obj
        counts[(label, position)] = counts.get((label, position), 0) + 1
    return tuple(sorted((label, position, n) for (label, position), n in counts.items()))


def _pattern_key_to_objects(key: PatternKey) -> List[dict]:
    return [{"label": label, "position": position, "count": n} for label, position, n in key]


def _pattern_key_from_objects(objects: List[dict]) -> PatternKey:
    return tuple(sorted((o["label"], o["position"], o.get("count", 1)) for o in objects))


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


class AudioOutput:
    """
    Priority speech queue backed by a warm, in-process Piper voice. A
    newly-queued item that outranks whatever is currently playing
    interrupts it immediately.

    Usage (pointing at an existing piper/ folder from a previous project):
        with AudioOutput(piper_dir="C:/path/to/piper") as audio:
            audio.speak("chair ahead", Priority.NARRATION)
            audio.speak("obstacle very close!", Priority.ALERT)  # interrupts

    Or, if you just want to name a model file directly:
        with AudioOutput(model_path="C:/path/to/voice.onnx") as audio:
            ...

    For phrases you know you'll speak often and that are time-critical
    (safety alerts, the emergency confirmation), call precache() once
    after start() so those specific strings are synthesized up front and
    play back instantly instead of paying synthesis latency the first
    time they're needed:
        audio.precache(["Stop! Obstacle very close", "Emergency alert triggered"])

    Anything ELSE spoken through speak() -- any narration sentence, one
    object or five -- is also memoized automatically the first time it's
    synthesized, so the cache keeps growing at runtime: the first time a
    given sentence is heard it pays full Piper synthesis latency, but
    every identical sentence after that (the same object pattern being
    seen again) is an instant cache hit, same as anything from
    precache(). Callers don't need to do anything special for this to
    happen -- it's just what speak() does.

    Pass `objects` to speak() (the Detections that produced `text`) and
    two more things happen: (1) the audio is filed under that object
    PATTERN, not just its exact text, so a later batch with the same set
    of objects reuses the cached audio even if it arrived as slightly
    different phrasing; (2) if `cache_dir` was given at construction, the
    audio and its {"objects": [...], "text": ..., "file": ...} entry are
    written to disk -- so restarting the Pi process does NOT lose what
    it's already learned. On the next boot, __init__ reloads every
    cached phrase (and every object pattern it maps to) straight off
    disk, before the main loop even starts. This is what makes the
    server's per-flush narration sentence (see server_webrtc.py and
    pi_webrtc_client.py's GatheredBatch) a real cache instead of a
    re-synthesis every 2 seconds, and keeps it that way across restarts:
    the same gathered object-set always composes to the same text
    (narration_common.template_narrate() is deterministic), so the next
    time that exact scene is seen -- today or next week -- speak() finds
    it already cached and skips Piper entirely.
    """

    @staticmethod
    def _resolve_piper_dir(piper_dir: Optional[str]) -> Optional[str]:
        if not piper_dir:
            return None

        candidate = _normalize_path(piper_dir)
        if os.path.isabs(piper_dir):
            return candidate

        candidates = [
            candidate,
            _normalize_path(os.path.join(os.getcwd(), piper_dir)),
            _normalize_path(os.path.join(os.path.dirname(__file__), piper_dir)),
            _normalize_path(os.path.join(os.path.dirname(__file__), "..", piper_dir)),
        ]
        for item in candidates:
            if os.path.isdir(item):
                return item
        return candidate

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
        if not model_path:
            return None
        if os.path.isabs(model_path):
            return model_path

        candidates = [
            _normalize_path(model_path),
            _normalize_path(os.path.join(os.getcwd(), model_path)),
            _normalize_path(os.path.join(os.path.dirname(__file__), model_path)),
            _normalize_path(os.path.join(os.path.dirname(__file__), "..", model_path)),
        ]
        for item in candidates:
            if os.path.exists(item):
                return item
        return _normalize_path(model_path)

    def __init__(
        self,
        piper_dir: str = None,
        model_path: str = None,
        use_cuda: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if piper_dir:
            piper_dir = self._resolve_piper_dir(piper_dir)
        if piper_dir and model_path is None:
            print(os.path.join(piper_dir, "voices"))
            model_path = self._find_voice_in_dir(os.path.join(piper_dir, "voices"))

        if model_path is not None:
            model_path = self._resolve_model_path(model_path)

        self._model_path = self._resolve_model(model_path)
        self._voice = self._load_voice(self._model_path, use_cuda)

        self._cache: Dict[str, bytes] = {}              # text -> wav bytes (in-memory, always populated)
        self._pattern_cache: Dict[PatternKey, str] = {}  # object pattern -> text (only for object narration)
        self._cache_lock = threading.Lock()  # guards _cache/_pattern_cache/disk writes -- precache() writes from the caller's thread, _synthesize_and_play() writes from the worker thread
        self._queue: "queue.PriorityQueue[_QueuedItem]" = queue.PriorityQueue()
        self._current_playback = None      # "winsound", a Popen, or None
        self._current_priority = -1
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Disk persistence: if given, every synthesized phrase is written
        # to <cache_dir>/<hash>.wav and recorded in <cache_dir>/manifest.json,
        # and both are loaded back into memory here at construction time --
        # so a Pi restart doesn't re-pay Piper synthesis for anything it has
        # ever spoken before. Without cache_dir, behavior is unchanged from
        # before: an in-memory-only cache that starts empty every run.
        self._cache_dir = cache_dir
        self._manifest_path = os.path.join(cache_dir, "manifest.json") if cache_dir else None
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
            self._load_disk_cache()

    def _load_disk_cache(self):
        """Loads manifest.json (list of {"text", "file", "objects"} entries)
        and the wav bytes each entry points at, populating both _cache
        (text -> audio) and _pattern_cache (object pattern -> text)."""
        if not os.path.isfile(self._manifest_path):
            return
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Couldn't read TTS cache manifest (%s) -- starting fresh: %s", self._manifest_path, e)
            return

        loaded, missing = 0, 0
        for entry in manifest:
            wav_path = os.path.join(self._cache_dir, entry["file"])
            try:
                with open(wav_path, "rb") as f:
                    wav_bytes = f.read()
            except OSError:
                missing += 1
                continue  # manifest references a file that's gone -- skip it, don't crash startup
            self._cache[entry["text"]] = wav_bytes
            if entry.get("objects"):
                self._pattern_cache[_pattern_key_from_objects(entry["objects"])] = entry["text"]
            loaded += 1

        logger.info(
            "Loaded %d cached TTS phrase(s) from disk (%s)%s -- no re-synthesis needed for any of them.",
            loaded, self._cache_dir, f", {missing} stale entr{'y' if missing == 1 else 'ies'} skipped" if missing else "",
        )

    def _save_to_disk(self, text: str, wav_bytes: bytes, objects: Optional[List[dict]] = None):
        """Must be called with self._cache_lock held. Writes the wav file
        (if not already on disk) and appends/updates its manifest entry."""
        if not self._cache_dir:
            return
        file_name = hashlib.sha1(text.encode("utf-8")).hexdigest()[:20] + ".wav"
        wav_path = os.path.join(self._cache_dir, file_name)
        if not os.path.isfile(wav_path):
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)

        manifest = []
        if os.path.isfile(self._manifest_path):
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                manifest = []

        for entry in manifest:
            if entry["text"] == text:
                if objects and not entry.get("objects"):
                    entry["objects"] = objects  # backfill pattern info if this text is now known to map to one
                break
        else:
            manifest.append({"text": text, "file": file_name, "objects": objects})

        tmp_path = self._manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, self._manifest_path)  # atomic swap -- never leaves a half-written manifest

    def _find_voice_in_dir(self, voices_dir: str):
        """Auto-picks the first .onnx file (with a matching .onnx.json) in voices_dir."""
        if not os.path.isdir(voices_dir):
            return None
        for candidate in sorted(glob.glob(os.path.join(voices_dir, "*.onnx"))):
            if os.path.isfile(candidate + ".json"):
                return candidate
        return None

    def _resolve_model(self, model_path: str) -> str:
        if model_path is None:
            model_path = os.environ.get("PIPER_VOICE_MODEL")
        if model_path is not None:
            model_path = self._resolve_model_path(model_path)
        if model_path is None:
            raise AudioError(
                "No Piper voice model found. Pass AudioOutput(piper_dir='path/to/piper') "
                "(auto-detects a voice in piper/voices/), or "
                "AudioOutput(model_path='path/to/voice.onnx') directly, or set the "
                "PIPER_VOICE_MODEL environment variable. If you don't have a voice yet, "
                "download one from https://huggingface.co/rhasspy/piper-voices (you need "
                "both the .onnx file and its matching .onnx.json config file)."
            )
        if not os.path.isfile(model_path):
            raise AudioError(f"Piper voice model not found: {model_path}")
        if not os.path.isfile(model_path + ".json"):
            logger.warning(
                "No config file found next to %s (expected %s.json) -- "
                "Piper may fail to load this voice.", model_path, model_path,
            )
        return model_path

    def _load_voice(self, model_path: str, use_cuda: bool):
        """Loads the ONNX voice model into memory ONCE. This is the whole
        point of the rewrite: everything after this (every speak() call,
        for the lifetime of the process) reuses this same in-memory
        model instead of reloading it from disk."""
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise AudioError(
                "The 'piper-tts' package isn't installed. Run `pip install piper-tts` "
                "(this replaces the old standalone `piper` CLI binary -- you no longer "
                "need to download/point at a separate piper executable)."
            ) from e

        logger.info("Loading Piper voice model from %s (one-time cost)...", model_path)
        t0 = time.monotonic()
        try:
            voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        except Exception as e:
            raise AudioError(f"Failed to load Piper voice model '{model_path}': {e}") from e
        logger.info(
            "Piper voice loaded in %.2fs -- model now stays warm in memory for all "
            "future speak() calls.", time.monotonic() - t0,
        )
        return voice

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("AudioOutput started (voice: %s)", self._model_path)

    def precache(self, phrases: Iterable[str]):
        """
        Synthesizes each phrase once, right now, and keeps the resulting
        audio in memory. A later speak() call with the EXACT same text
        skips synthesis entirely and plays the cached audio straight
        away -- use this for phrases you already know you'll need (fixed
        safety phrases, single-object narration) so they never pay
        synthesis latency at the moment they're needed.

        Call this after start(), before the main loop begins.
        """
        for phrase in phrases:
            with self._cache_lock:
                already_cached = phrase in self._cache
            if already_cached:
                continue
            t0 = time.monotonic()
            wav_bytes = self._synthesize_to_bytes(phrase)
            with self._cache_lock:
                self._cache[phrase] = wav_bytes
                self._save_to_disk(phrase, wav_bytes)
            logger.info("Pre-cached TTS phrase (%.2fs): %r", time.monotonic() - t0, phrase)

    def speak(self, text: str, priority: "Priority" = Priority.NARRATION, objects: Optional[Iterable] = None):
        """
        Queues `text` to be spoken. If `objects` is given (the list of
        Detections -- or (label, position) pairs/dicts -- that this exact
        text narrates), the audio also gets filed under that object
        pattern: {"objects": [...], "text": ..., "file": ...} in the
        on-disk manifest. That's what lets a later call check "have I
        already got audio for THIS SET of objects" directly, and it's
        also what survives a Pi restart (see cache_dir / _load_disk_cache).

        Only pass `objects` when the text is a deterministic function of
        that object set (the server's flush sentence or a locally
        template_narrate()'d one) -- not for one-off phrasings like an
        Ollama-composed sentence, which can differ each time for the same
        objects and shouldn't overwrite the pattern's canonical audio.
        """
        pattern_key = make_pattern_key(objects) if objects is not None else None
        item = _QueuedItem(
            sort_key=-int(priority), timestamp=time.time(), text=text, priority=int(priority),
            pattern_key=pattern_key,
        )
        self._queue.put(item)
        with self._lock:
            if self._current_playback is not None and int(priority) > self._current_priority:
                logger.info("Interrupting current speech for higher-priority item: %r", text)
                self._stop_playback()

    def has_cached_pattern(self, objects: Iterable) -> bool:
        """True if this exact object pattern (regardless of text wording)
        already has cached audio -- from this session or a prior one."""
        with self._cache_lock:
            return make_pattern_key(objects) in self._pattern_cache

    def _worker_loop(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._synthesize_and_play(item)

    def _synthesize_and_play(self, item: "_QueuedItem"):
        with self._lock:
            self._current_priority = item.priority

        try:
            with self._cache_lock:
                # Pattern match first: if this exact set of objects has
                # been spoken before (possibly under slightly different
                # wording, or in a previous run before a restart), reuse
                # that cached audio/text directly instead of re-checking
                # by text -- this is the {[objects] -> narration audio}
                # lookup, backed by the text->bytes cache underneath it.
                text = item.text
                if item.pattern_key is not None and item.pattern_key in self._pattern_cache:
                    text = self._pattern_cache[item.pattern_key]
                wav_bytes = self._cache.get(text)

            if wav_bytes is None:
                # Cache miss -- this exact sentence (usually a
                # multi-object narration composed by
                # narration_common.template_narrate(), either locally
                # or on the server) hasn't been heard before, on disk or
                # in memory. Synthesize it now and memoize the result
                # under its exact text (and, if given, its object
                # pattern) so the next time this same batch is gathered
                # (see server_webrtc.py's process_video() /
                # pi_webrtc_client.GatheredBatch) -- this run or after a
                # restart -- it's an instant replay instead of paying
                # Piper synthesis again.
                t0 = time.monotonic()
                wav_bytes = self._synthesize_to_bytes(text)
                objects = _pattern_key_to_objects(item.pattern_key) if item.pattern_key is not None else None
                with self._cache_lock:
                    self._cache[text] = wav_bytes
                    if item.pattern_key is not None:
                        self._pattern_cache[item.pattern_key] = text
                    self._save_to_disk(text, wav_bytes, objects)
                logger.info(
                    "Learned new phrase (%.2fs), now cached (disk: %s) for next time: %r",
                    time.monotonic() - t0, bool(self._cache_dir), text,
                )
            self._play_and_wait(wav_bytes)
        except Exception as e:
            logger.error("Playback error: %s", e)
        finally:
            with self._lock:
                self._current_playback = None
                self._current_priority = -1

    def _synthesize_to_bytes(self, text: str) -> bytes:
        """Runs the already-loaded Piper voice to synthesize `text` to an
        in-memory WAV buffer -- no disk I/O, no process spawn, no model
        reload. Only the actual inference cost is paid here."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return buf.getvalue()

    def _play_and_wait(self, wav_bytes: bytes):
        if SYSTEM == "Windows":
            import winsound
            with self._lock:
                self._current_playback = "winsound"
            # SND_MEMORY plays straight from the in-memory WAV bytes -- no temp file.
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
        elif SYSTEM == "Darwin":
            # afplay has no stdin-streaming mode, so macOS (dev-only in this
            # project) still needs a real file on disk; Linux/Pi below does not.
            import tempfile
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(wav_bytes)
                proc = subprocess.Popen(["afplay", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with self._lock:
                    self._current_playback = proc
                proc.wait()
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
        else:
            # Linux/Pi: stream the WAV bytes straight into aplay's stdin --
            # no temp file, no extra disk I/O in the hot path.
            proc = subprocess.Popen(
                ["aplay", "-q"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            with self._lock:
                self._current_playback = proc
            try:
                proc.stdin.write(wav_bytes)
                proc.stdin.close()
            except BrokenPipeError:
                pass  # aplay was killed (interrupted by a higher-priority item)
            proc.wait()

    def _stop_playback(self):
        """Must be called with self._lock held."""
        if self._current_playback == "winsound":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif self._current_playback is not None:
            self._current_playback.kill()

    def stop(self):
        self._running = False
        with self._lock:
            self._stop_playback()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("AudioOutput stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    print("Testing AudioOutput with a warm, in-process Piper voice\n")

    # Usage: python pi_audio.py [path/to/piper_dir]
    # If omitted, falls back to PIPER_VOICE_MODEL env var.
    piper_dir = sys.argv[1] if len(sys.argv) > 1 else None

    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "./tts_cache"

    try:
        with AudioOutput(piper_dir=piper_dir, cache_dir=cache_dir) as audio:
            audio.precache(["Obstacle ahead!"])
            print("\n-- First 'Obstacle ahead!' call below is served from cache (should be instant) --")
            audio.speak("Scene narration test, low priority.", Priority.NARRATION)
            time.sleep(0.5)
            audio.speak("Obstacle ahead!", Priority.ALERT)  # should interrupt narration, and be cached
            time.sleep(3)

            objects = [{"label": "chair", "position": "center"}, {"label": "person", "position": "center"},
                       {"label": "person", "position": "center"}, {"label": "tv", "position": "left"}]
            print("\n-- Cache-miss then cache-hit demo, keyed by OBJECT PATTERN --")
            audio.speak("chair and 2 persons ahead, tv to your left", Priority.NARRATION, objects=objects)
            time.sleep(3)
            print(f"  has_cached_pattern(objects) -> {audio.has_cached_pattern(objects)} (should be True now)")
            audio.speak("chair and 2 persons ahead, tv to your left", Priority.NARRATION, objects=objects)
            time.sleep(3)

            print(f"\n-- Cache saved to {cache_dir}/manifest.json --")
            print("  Run this script again with the same cache_dir argument: the phrases above")
            print("  will be an instant cache hit with zero Piper calls, even on a fresh process.")
        print("OK -- audio playback working.")
    except AudioError as e:
        print(f"Audio error: {e}")