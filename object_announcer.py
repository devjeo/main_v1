"""
Object Announcer -- decides WHAT gets spoken, and when.

Pure logic, no camera / YOLO / audio, so it can be tested anywhere.

Rules
-----
1. GATHER: every detection is collected, keyed by its tracker ID
   (Detection.track_id) -- seeing the same chair in 40 frames is still one
   chair. The newest sighting wins, so label/position reflect the object's
   latest state. The time of each sighting is remembered too.
2. SETTLE (debounce): the announcer waits until the scene has "settled" --
   no NEW announceable object has appeared for a short time -- and then
   speaks everything gathered in ONE batch.
     * Urgent objects (cars, bikes, a person straight ahead...) use a SHORT
       settle (default 0.5 s) so warnings are not delayed.
     * Everything else uses settle_s (default 1.5 s).
     * A hard cap (max_wait_s, default 6 s) guarantees something is spoken
       even if new objects keep trickling in.
3. MIN GAP: non-urgent announcements are at least min_gap_s (default 3 s)
   apart, so the cane speaks in a calm rhythm instead of interrupting itself.
   URGENT objects ignore the gap -- a warning never waits.
4. STALE FILTER: at flush time, objects not seen for more than stale_s
   (default 1 s) are dropped. They already left the camera view (you walked
   past them), so announcing them would be misleading.
5. COOLDOWN: an ID announced less than cooldown_s (default 10 s) ago is not
   "new" and does not trigger a flush. It is reported in
   WindowResult.suppressed so the caller can still include it in the
   sentence (so counts describe the WHOLE scene, not just the newcomers).
   The cooldown is a fixed 10 s from the moment of announcement
   (continuing to see the object does NOT extend it).

Because the cooldown is per ID and not per label, two phones are two
objects: phone #1 stays quiet while phone #2 (a new ID) is announced.

Detections whose track_id is None (tracker hasn't confirmed them yet) are
ignored -- without an ID we can't tell "same object" from "new object".

When nothing announceable is pending, the gathered set is simply cleared
every window_s seconds (housekeeping) so vanished objects drop out.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set

from detection_common import Detection

logger = logging.getLogger("object_announcer")

# Labels that get the short settle time and ignore the min gap. A "person"
# is only urgent when straight ahead (see default_is_urgent).
URGENT_LABELS = frozenset({
    "car", "truck", "motorcycle", "bicycle", "bus", "train",
    "dog", "horse", "cow", "sheep",
})


def default_is_urgent(det: Detection) -> bool:
    if det.label in URGENT_LABELS:
        return True
    return det.label == "person" and det.position == "center"


@dataclass
class WindowResult:
    announce: List[Detection] = field(default_factory=list)    # new objects -> trigger the speech
    suppressed: List[Detection] = field(default_factory=list)  # still in view, announced < cooldown ago


class ObjectAnnouncer:
    def __init__(
        self,
        window_s: float = 3.0,
        cooldown_s: float = 10.0,
        settle_s: float = 1.5,
        urgent_settle_s: float = 0.5,
        max_wait_s: float = 6.0,
        min_gap_s: float = 3.0,
        stale_s: float = 1.0,
        is_urgent: Callable[[Detection], bool] = default_is_urgent,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.window_s = window_s              # housekeeping interval when nothing is pending
        self.cooldown_s = cooldown_s
        self.settle_s = settle_s
        self.urgent_settle_s = urgent_settle_s
        self.max_wait_s = max_wait_s
        self.min_gap_s = min_gap_s            # minimum time between NON-urgent announcements
        self.stale_s = stale_s                # objects unseen for longer than this are not announced
        self.is_urgent = is_urgent
        self.clock = clock

        self._window_start: Optional[float] = None
        self._seen: Dict[int, Detection] = {}          # track_id -> latest Detection
        self._last_seen: Dict[int, float] = {}         # track_id -> time of latest sighting
        self._pending: Set[int] = set()                # announceable IDs gathered so far
        self._first_pending: Optional[float] = None
        self._last_new: Optional[float] = None         # last time any announceable ID joined
        self._last_new_urgent: Optional[float] = None  # last time an urgent one joined
        self._last_announced: Dict[int, float] = {}    # track_id -> time it was last spoken
        self._last_flush: Optional[float] = None       # last time something was spoken (NOT cleared in _reset)

    def _ensure_window(self) -> None:
        if self._window_start is None:
            self._window_start = self.clock()

    def _announceable(self, track_id: int, now: float) -> bool:
        last = self._last_announced.get(track_id)
        return last is None or now - last >= self.cooldown_s

    def _is_stale(self, track_id: int, now: float) -> bool:
        return now - self._last_seen.get(track_id, now) > self.stale_s

    def add(self, detections: Iterable[Detection]) -> None:
        """Feed every frame's detections in. Cheap; call as often as you like."""
        self._ensure_window()
        now = self.clock()
        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            self._seen[tid] = det
            self._last_seen[tid] = now
            if tid not in self._pending and self._announceable(tid, now):
                self._pending.add(tid)
                if self._first_pending is None:
                    self._first_pending = now
                self._last_new = now
                if self.is_urgent(det):
                    self._last_new_urgent = now

    def _should_flush(self, now: float) -> bool:
        if now - self._first_pending >= self.max_wait_s:
            return True
        # Urgent objects: short settle, and they ignore the min gap.
        if any(self.is_urgent(self._seen[t]) for t in self._pending):
            ref = self._last_new_urgent if self._last_new_urgent is not None else self._last_new
            if now - ref >= self.urgent_settle_s:
                return True
        # Everything else: wait for the scene to settle AND for the min gap.
        gap_ok = self._last_flush is None or now - self._last_flush >= self.min_gap_s
        return gap_ok and now - self._last_new >= self.settle_s

    def _reset(self, now: float) -> None:
        self._seen = {}
        self._last_seen = {}
        self._pending = set()
        self._first_pending = None
        self._last_new = None
        self._last_new_urgent = None
        self._window_start = now

    def poll(self) -> Optional[WindowResult]:
        """
        Call every loop iteration. Returns None until there is something to
        report. A WindowResult with a non-empty `announce` means: speak now.
        (announce is empty only for the periodic housekeeping result, or when
        every pending object had already left the view.)
        """
        self._ensure_window()
        now = self.clock()

        if not self._pending:
            if now - self._window_start < self.window_s:
                return None
            result = WindowResult(
                suppressed=[d for t, d in self._seen.items() if not self._is_stale(t, now)])
            self._reset(now)
            return result

        if not self._should_flush(now):
            return None

        result = WindowResult()
        for tid, det in self._seen.items():
            if self._is_stale(tid, now):
                continue                      # already left the view, don't announce it
            if tid in self._pending:
                result.announce.append(det)
                self._last_announced[tid] = now
            else:
                result.suppressed.append(det)

        if result.announce:
            self._last_flush = now            # only a real announcement starts the min gap

        # Forget IDs whose cooldown has expired so memory can't grow forever.
        self._last_announced = {
            tid: ts for tid, ts in self._last_announced.items() if now - ts < self.cooldown_s
        }
        self._reset(now)
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

    def mk(label, track_id, position="left"):
        return Detection(label=label, confidence=0.9, bbox=[0, 0, 1, 1],
                         position=position, track_id=track_id)

    def new_run():
        clock = FakeClock()
        ann = ObjectAnnouncer(window_s=3.0, cooldown_s=10.0, settle_s=1.5,
                              urgent_settle_s=0.5, max_wait_s=6.0,
                              min_gap_s=3.0, stale_s=1.0, clock=clock)
        log = []

        def run_until(t_end, scene_fn):
            """Advance in 0.5 s steps; scene_fn(t) -> detections for that moment."""
            while clock.t < t_end:
                clock.t = round(clock.t + 0.5, 3)
                ann.add(scene_fn(clock.t))
                res = ann.poll()
                if res is not None and (res.announce or res.suppressed):
                    log.append((clock.t,
                                sorted(d.track_id for d in res.announce),
                                sorted(d.track_id for d in res.suppressed)))
        return clock, ann, log, run_until

    def spoken(log):
        """Only the entries where something was actually announced."""
        return [e for e in log if e[1]]

    # 1) Growing group of non-urgent objects is batched into ONE announcement.
    _, _, log, run = new_run()
    chairs = lambda t: [mk("chair", 1)] + ([mk("chair", 2)] if t >= 1.5 else []) + \
                       ([mk("chair", 3)] if t >= 2.5 else [])
    run(5.0, chairs)
    assert spoken(log) == [(4.0, [1, 2, 3], [])], log   # settles 1.5 s after the LAST newcomer
    print("OK 1: growing group batched ->", spoken(log)[0])

    # 2) An urgent object (car) is spoken after only ~0.5 s.
    _, _, log, run = new_run()
    run(2.0, lambda t: [mk("car", 7)])
    assert spoken(log) == [(1.0, [7], [])], log
    print("OK 2: urgent car announced fast ->", spoken(log)[0])

    # 3) Second wave: earlier IDs are suppressed but still reported, so the
    #    caller can speak announce+suppressed = the whole scene.
    _, _, log, run = new_run()
    def scene(t):
        s = [mk("person", 1, "left"), mk("person", 2, "left")]
        if t >= 6.0:
            s += [mk("person", 3, "left"), mk("person", 4, "left")]
        return s
    run(9.0, scene)
    assert spoken(log) == [(2.0, [1, 2], []), (7.5, [3, 4], [1, 2])], log
    print("OK 3: second wave ->", spoken(log)[1], "(caller speaks all 4)")

    # 4) Cooldown expiry: the same ID may be spoken again after 10 s.
    _, _, log, run = new_run()
    run(16.0, lambda t: [mk("tv", 5)])
    s = spoken(log)
    assert [e[1] for e in s] == [[5], [5]], log
    assert s[1][0] - s[0][0] >= 10.0, log
    print("OK 4: repeat after cooldown ->", [e[0] for e in s])

    # 5) Unconfirmed (track_id=None) detections are ignored.
    _, _, log, run = new_run()
    run(8.0, lambda t: [Detection("chair", 0.8, [0, 0, 1, 1], "center", track_id=None)])
    assert not spoken(log), log
    print("OK 5: unconfirmed detections ignored")

    # 6) Hard cap: a never-ending trickle of newcomers still gets spoken.
    _, _, log, run = new_run()
    run(7.0, lambda t: [mk("chair", i) for i in range(1, int(t * 1) + 2)])
    assert spoken(log) and spoken(log)[0][0] <= 6.5, log   # first newcomer at 0.5 s + 6 s cap
    print("OK 6: max_wait cap ->", spoken(log)[0])

    # 7) Stale filter: chair 1 leaves the view right away, chair 2 stays.
    #    Only chair 2 is spoken; the one already passed stays silent.
    _, _, log, run = new_run()
    run(3.0, lambda t: ([mk("chair", 1)] if t <= 0.5 else []) + [mk("chair", 2)])
    assert spoken(log) == [(2.0, [2], [])], log
    print("OK 7: object that left the view is not announced ->", spoken(log)[0])

    # 8) Min gap: a newcomer right after a flush waits until 3 s have passed.
    _, _, log, run = new_run()
    run(6.0, lambda t: [mk("chair", 1)] + ([mk("chair", 2)] if t >= 2.5 else []))
    assert spoken(log) == [(2.0, [1], []), (5.0, [2], [1])], log   # settled at 4.0, held until 5.0
    print("OK 8: min gap holds the second announcement ->", spoken(log)[1])

    # 9) Urgent objects ignore the min gap.
    _, _, log, run = new_run()
    run(4.0, lambda t: [mk("chair", 1)] + ([mk("car", 9, "center")] if t >= 2.5 else []))
    assert spoken(log) == [(2.0, [1], []), (3.0, [9], [1])], log
    print("OK 9: urgent car skips the gap ->", spoken(log)[1])

    print("\nAll announcer checks passed.")