"""
Narration Common -- dependency-free template sentence composer.

Used on BOTH machines, for different reasons:
  - On the Pi: this IS the narration path whenever local fallback
    detection is active (no server, no network, no LLM -- matches
    PROJECT.md's requirement that local fallback stays fully offline).
  - On the laptop server: this is what server_webrtc.py's process_video()
    calls to compose the narration sentence it bundles into every
    2-second flush (see GatheredBatch in pi_webrtc_client.py), and it's
    also server_narrator.py's own fallback if Ollama itself is
    unreachable/fails, so server-mode narration always degrades
    gracefully instead of going silent.

No external dependencies (no `ollama` import) -- safe to have on the Pi
without pulling in anything LLM-related.
"""

from collections import defaultdict
from typing import Dict, List

from detection_common import Detection


def template_narrate(detections: List[Detection]) -> str:
    """
    Groups detections by position, counts duplicate labels within each
    group, and joins into one sentence.

    e.g. [mouse/center, tv/left, laptop/left, person x3/center]
      -> "mouse and 3 persons ahead, tv and laptop to your left"

    Labels within a position are sorted alphabetically (not left in
    whatever order they happened to arrive) so the exact same SET of
    objects always produces the exact same sentence text, regardless of
    which order YOLO reported them in this time. That determinism is
    what makes this function's output usable as a cache key elsewhere
    (see pi_audio.py's AudioOutput, which memoizes synthesized audio by
    exact text match) -- if arrival order could shuffle the wording,
    the same real-world scene could produce two different strings and
    never hit the cache. It's also what lets the server compose this
    sentence once at flush time (server_webrtc.py) and hand it to the
    Pi ready-to-speak: the same object set always yields the same text
    on both machines, so nothing has to be recomputed or re-requested.
    """
    if not detections:
        return ""

    by_position: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in detections:
        by_position[d.position][d.label] += 1

    parts = []
    for position in ("center", "left", "right"):
        if position not in by_position:
            continue
        labels = [
            (label if count == 1 else f"{count} {label}s")
            for label, count in sorted(by_position[position].items())
        ]
        joined = " and ".join(labels)
        direction = "ahead" if position == "center" else f"to your {position}"
        parts.append(f"{joined} {direction}")

    return ", ".join(parts)


if __name__ == "__main__":
    print("Testing template_narrate (no dependencies required)\n")
    detections = [
        Detection(label="mouse", confidence=0.9, bbox=[0, 0, 1, 1], position="center"),
        Detection(label="tv", confidence=0.9, bbox=[0, 0, 1, 1], position="left"),
        Detection(label="laptop", confidence=0.9, bbox=[0, 0, 1, 1], position="left"),
        Detection(label="person", confidence=0.9, bbox=[0, 0, 1, 1], position="center"),
        Detection(label="person", confidence=0.9, bbox=[0, 0, 1, 1], position="left"),
        Detection(label="person", confidence=0.9, bbox=[0, 0, 1, 1], position="right"),
    ]
    print(template_narrate(detections))

    print("\nOrder-independence check (same set, different arrival order):")
    shuffled = [detections[3], detections[0], detections[5], detections[1], detections[4], detections[2]]
    a, b = template_narrate(detections), template_narrate(shuffled)
    print(f"  {a!r}")
    print(f"  {b!r}")
    print("  MATCH" if a == b else "  MISMATCH -- cache-breaking bug!")