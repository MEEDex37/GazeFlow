"""Gaze direction and screen coordinate estimation."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque

from src.gazeflow.eye_tracker import EyeLandmarks, Point

VERTICAL_CENTER_BIAS = 0.006


@dataclass(frozen=True)
class GazeResult:
    """Coarse gaze direction with supporting eye-position ratios."""

    direction: str
    horizontal_ratio: float
    vertical_ratio: float

    @property
    def label(self) -> str:
        """Return a user-facing gaze label."""
        labels = {
            "left": "Looking Left",
            "right": "Looking Right",
            "center": "Looking Center",
            "up": "Looking Up",
            "down": "Looking Down",
            "unknown": "Gaze Unknown",
        }
        return labels.get(self.direction, labels["unknown"])


class GazeEstimator:
    """Estimate coarse gaze direction from eye and iris landmarks."""

    def __init__(
        self,
        history_size: int = 3,
        horizontal_low: float = 0.45,
        horizontal_high: float = 0.55,
        vertical_low: float = 0.46,
        vertical_high: float = 0.54,
        mirror_horizontal: bool = False,
        vertical_top: float | None = None,
        vertical_center: float | None = None,
        vertical_bottom: float | None = None,
    ) -> None:
        self.horizontal_low = horizontal_low
        self.horizontal_high = horizontal_high
        self.vertical_low = vertical_low
        self.vertical_high = vertical_high
        self.mirror_horizontal = mirror_horizontal
        self.vertical_top = vertical_top
        self.vertical_center = vertical_center
        self.vertical_bottom = vertical_bottom
        self._history: Deque[str] = deque(maxlen=history_size)

    def estimate(self, eye_landmarks: EyeLandmarks) -> GazeResult:
        """Return a smoothed coarse gaze direction."""
        left_ratio = self._eye_ratio(eye_landmarks.left_eye, eye_landmarks.left_iris)
        right_ratio = self._eye_ratio(eye_landmarks.right_eye, eye_landmarks.right_iris)

        if left_ratio is None and right_ratio is None:
            direction = "unknown"
            horizontal_ratio = 0.5
            vertical_ratio = 0.5
        else:
            ratios = [ratio for ratio in (left_ratio, right_ratio) if ratio is not None]
            horizontal_ratio = sum(ratio[0] for ratio in ratios) / len(ratios)
            vertical_ratio = sum(ratio[1] for ratio in ratios) / len(ratios)
            direction = self._classify(horizontal_ratio, vertical_ratio)

        self._history.append(direction)
        return GazeResult(
            direction=self._smoothed_direction(),
            horizontal_ratio=horizontal_ratio,
            vertical_ratio=vertical_ratio,
        )

    def estimate_direction(self, eye_landmarks: EyeLandmarks) -> str:
        """Return only the smoothed direction string."""
        return self.estimate(eye_landmarks).direction

    def _eye_ratio(self, eye_points: list[Point], iris_points: list[Point]) -> tuple[float, float] | None:
        if not eye_points or not iris_points:
            return None

        min_x = min(point[0] for point in eye_points)
        max_x = max(point[0] for point in eye_points)
        min_y = min(point[1] for point in eye_points)
        max_y = max(point[1] for point in eye_points)
        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            return None

        iris_x = sum(point[0] for point in iris_points) / len(iris_points)
        iris_y = sum(point[1] for point in iris_points) / len(iris_points)

        return ((iris_x - min_x) / width, (iris_y - min_y) / height)

    def _classify(self, horizontal_ratio: float, vertical_ratio: float) -> str:
        if self.mirror_horizontal:
            horizontal_ratio = 1.0 - horizontal_ratio

        horizontal_direction = self._classify_horizontal(horizontal_ratio)
        vertical_direction = self._classify_vertical(vertical_ratio)
        horizontal_delta = abs(horizontal_ratio - 0.5)
        vertical_delta = self._vertical_delta(vertical_ratio)

        if vertical_direction in {"up", "down"} and vertical_delta > horizontal_delta * 1.05:
            return vertical_direction

        if horizontal_direction in {"left", "right"}:
            return horizontal_direction
        if vertical_direction in {"up", "down"}:
            return vertical_direction

        return "center"

    def _classify_horizontal(self, horizontal_ratio: float) -> str:
        if horizontal_ratio < self.horizontal_low:
            return "left"
        if horizontal_ratio > self.horizontal_high:
            return "right"

        return "center"

    def _classify_vertical(self, vertical_ratio: float) -> str:
        if self._has_vertical_calibration():
            distances = {
                "up": abs(vertical_ratio - self.vertical_top),
                "center": abs(vertical_ratio - self.vertical_center),
                "down": abs(vertical_ratio - self.vertical_bottom),
            }
            direction = min(distances, key=distances.get)
            if (
                direction != "center"
                and distances["center"] <= distances[direction] + VERTICAL_CENTER_BIAS
            ):
                return "center"

            return direction

        if vertical_ratio < self.vertical_low:
            return "up"
        if vertical_ratio > self.vertical_high:
            return "down"

        return "center"

    def _vertical_delta(self, vertical_ratio: float) -> float:
        if self.vertical_center is not None:
            return abs(vertical_ratio - self.vertical_center)

        return abs(vertical_ratio - 0.5)

    def _has_vertical_calibration(self) -> bool:
        return (
            self.vertical_top is not None
            and self.vertical_center is not None
            and self.vertical_bottom is not None
        )

    def _smoothed_direction(self) -> str:
        if not self._history:
            return "unknown"

        counts = Counter(self._history)
        return counts.most_common(1)[0][0]
