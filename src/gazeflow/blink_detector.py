"""Blink detection from eye landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from src.gazeflow.eye_tracker import EyeLandmarks, Point


@dataclass(frozen=True)
class BlinkResult:
    """Blink event and eye openness state for one video frame."""

    blink_detected: bool
    is_closed: bool
    openness_ratio: float
    blink_count: int
    cooldown_remaining: int


class BlinkDetector:
    """Detect blinks from eye contour openness."""

    def __init__(self, closed_threshold: float = 0.18, cooldown_frames: int = 8) -> None:
        self.closed_threshold = closed_threshold
        self.cooldown_frames = cooldown_frames
        self.blink_count = 0
        self._was_closed = False
        self._cooldown_remaining = 0

    def detect(self, eye_landmarks: EyeLandmarks) -> BlinkResult:
        """Return whether the current frame contains a new blink event."""
        openness_ratio = self._average_openness(eye_landmarks)
        is_closed = openness_ratio < self.closed_threshold
        blink_detected = False

        if is_closed and not self._was_closed and self._cooldown_remaining == 0:
            blink_detected = True
            self.blink_count += 1
            self._cooldown_remaining = self.cooldown_frames
        elif self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        self._was_closed = is_closed

        return BlinkResult(
            blink_detected=blink_detected,
            is_closed=is_closed,
            openness_ratio=openness_ratio,
            blink_count=self.blink_count,
            cooldown_remaining=self._cooldown_remaining,
        )

    def reset(self) -> None:
        """Clear blink state and counters."""
        self.blink_count = 0
        self._was_closed = False
        self._cooldown_remaining = 0

    def _average_openness(self, eye_landmarks: EyeLandmarks) -> float:
        ratios = [
            ratio
            for ratio in (
                self._eye_openness(eye_landmarks.left_eye),
                self._eye_openness(eye_landmarks.right_eye),
            )
            if ratio is not None
        ]

        if not ratios:
            return 1.0

        return sum(ratios) / len(ratios)

    def _eye_openness(self, eye_points: list[Point]) -> float | None:
        if len(eye_points) < 16:
            return None

        eye_width = dist(eye_points[0], eye_points[8])
        if eye_width <= 0:
            return None

        vertical_distances = (
            dist(eye_points[1], eye_points[15]),
            dist(eye_points[2], eye_points[14]),
            dist(eye_points[3], eye_points[13]),
            dist(eye_points[4], eye_points[12]),
        )

        return (sum(vertical_distances) / len(vertical_distances)) / eye_width
