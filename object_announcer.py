"""
Object Announcer -- decides WHAT gets spoken, and when.

Pure logic, no camera / YOLO / audio, so it can be tested anywhere.

Rules
-----
1. GATHER: every detection is collected for a fixed window (default 3 s).
   Inside a window each physical object counts once, keyed by its tracker
   ID (Detection.track_id) -- seeing the same chair in 40 frames is still
   one chair. The newest sighting wins, so the label/position reflect the
   object's latest state.
2. SPEAK: when the window ends, every object in it is checked against a
   per-ID cooldown (default 10 s). An ID that was announced less than
   10 s ago is held back; everything else is announced, and its ID's
   cooldown starts now.
3. RESET: the cooldown is a fixed 10 s from the moment of announcement
   (continuing to see the object does NOT extend it). After that, the
   same ID can be spoken again if it is still in view.

Because the cooldown is per ID and not per label, two phones are two
objects: phone #1 announced in window 1 stays quiet while phone #2 (a new
ID) is announced in window 2.

Detections whose track_id is None (tracker hasn't confirmed them yet) are
ignored -- without an ID we can't tell "same object" from "new object",
and a confirmed ID usually arrives within a frame or two.

Granularity note: cooldowns are only checked when a window closes, so the
real gap before a repeat is the first window boundary at or after 10 s
(with 3 s windows: between 10 and 12 s).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from detection_common import Detection

logger = logging.getLogger("object_announcer")


@dataclass
class WindowResult:
    announce: List[Detection] = field(default_factory=list)    # new objects -> speak these
    suppressed: List[Detection] = field(default_factory=list)  # seen, but announced < cooldown ago


class ObjectAnnouncer:
    def __init__(
        self,
        window_s: float = 3.0,
        cooldown_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._window_start: Optional[float] = None
        self._seen: Dict[int, Detection] = {}          # track_id -> latest Detection this window
        self._last_announced: Dict[int, float] = {}    # track_id -> time it was last spoken

    def _ensure_window(self) -> None:
        if self._window_start is None:
            self._window_start = self.clock()

    def add(self, detections: List[Detection]) -> None:
        """Feed every frame's detections in. Cheap; call as often as you like."""
        self._ensure_window()
        for det in detections:
            if det.track_id is None:
                continue
            self._seen[det.track_id] = det

    def poll(self) -> Optional[WindowResult]:
        """
        Call every loop iteration. Returns None while the window is still
        open; once it has run for window_s, closes it and returns a
        WindowResult (announce may be empty -- nothing new this window).
        """
        self._ensure_window()
        now = self.clock()
        if now - self._window_start < self.window_s:
            return None

        result = WindowResult()
        for track_id, det in self._seen.items():
            last = self._last_announced.get(track_id)
            if last is not None and now - last < self.cooldown_s:
                result.suppressed.append(det)
            else:
                result.announce.append(det)
                self._last_announced[track_id] = now

        # Forget IDs whose cooldown has expired so memory can't grow forever.
        self._last_announced = {
            tid: ts for tid, ts in self._last_announced.items() if now - ts < self.cooldown_s
        }
        self._seen = {}
        self._window_start = now
        return result


# ----------------------------------------------------------------------
# Self-test with a fake clock: run `python object_announcer.py`
# ----------------------------------------------------------------------
if __name__ == "__main__":
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    def phone(track_id, position):
        return Detection(label="cell phone", confidence=0.9, bbox=[0, 0, 1, 1],
                         position=position, track_id=track_id)

    clock = FakeClock()
    ann = ObjectAnnouncer(window_s=3.0, cooldown_s=10.0, clock=clock)
    log = []

    def run_until(t_end, scene):
        """Advance the fake clock in 0.5 s steps, feeding `scene` each step."""
        while clock.t < t_end:
            clock.t = round(clock.t + 0.5, 3)
            ann.add(scene)
            res = ann.poll()
            if res is not None:
                log.append((clock.t,
                            sorted(d.track_id for d in res.announce),
                            sorted(d.track_id for d in res.suppressed)))

    phone_a = phone(1, "center")
    phone_b = phone(2, "left")
    ann.add([])  # open the first window at t=0

    run_until(3.0, [phone_a])               # window 1: only phone A
    run_until(6.0, [phone_a, phone_b])      # window 2: A still there, B added
    run_until(9.0, [phone_a, phone_b])      # window 3: nothing new
    run_until(12.0, [phone_a, phone_b])     # window 4: both still in cooldown
    run_until(15.0, [phone_a, phone_b])     # window 5: A's 10 s is up
    run_until(18.0, [phone_a, phone_b])     # window 6: B's 10 s is up
    run_until(21.0, [Detection("chair", 0.8, [0, 0, 1, 1], "center", track_id=None)])  # unconfirmed

    print(f"{'t':>5}  {'announce IDs':<14} suppressed IDs")
    for t, announced, suppressed in log:
        print(f"{t:>5}  {str(announced):<14} {suppressed}")

    expected = [
        (3.0, [1], []),        # phone A announced
        (6.0, [2], [1]),       # only the NEW phone B; A held back
        (9.0, [], [1, 2]),     # nothing new
        (12.0, [], [1, 2]),    # still inside both cooldowns
        (15.0, [1], [2]),      # A resets after 10 s and may speak again
        (18.0, [2], [1]),      # B resets (A is now in its fresh cooldown)
        (21.0, [], []),        # unconfirmed (no ID) detection is ignored
    ]
    assert log == expected, f"unexpected sequence:\n{log}\nexpected:\n{expected}"
    print("\nOK -- gather / per-ID dedup / 10 s reset all behave as specified.")