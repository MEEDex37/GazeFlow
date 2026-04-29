"""Calibration flow and calibration data storage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.gazeflow.gaze_estimator import GazeResult


CALIBRATION_FILE = Path("data/calibration_data.csv")
CALIBRATION_POINTS = (
    ("Top", 0.5, 0.15),
    ("Left", 0.15, 0.5),
    ("Center", 0.5, 0.5),
    ("Right", 0.85, 0.5),
    ("Bottom", 0.5, 0.85),
)
CSV_FIELDS = ("point_name", "screen_x", "screen_y", "eye_ratio_x", "eye_ratio_y")
DEFAULT_HORIZONTAL_LOW = 0.45
DEFAULT_HORIZONTAL_HIGH = 0.55
DEFAULT_VERTICAL_LOW = 0.46
DEFAULT_VERTICAL_HIGH = 0.54
MIN_AXIS_SEPARATION = 0.03
FALLBACK_CENTER_MARGIN = 0.035


@dataclass(frozen=True)
class CalibrationSample:
    """A single gaze-ratio sample captured for a calibration target."""

    point_name: str
    screen_x: float
    screen_y: float
    eye_ratio_x: float
    eye_ratio_y: float


@dataclass(frozen=True)
class CalibrationProfile:
    """Thresholds derived from saved calibration samples."""

    horizontal_low: float
    horizontal_high: float
    vertical_low: float
    vertical_high: float
    vertical_top: float | None = None
    vertical_center: float | None = None
    vertical_bottom: float | None = None


class CalibrationSession:
    """Collect and persist 5-point gaze calibration samples."""

    def __init__(
        self,
        output_path: Path = CALIBRATION_FILE,
        samples_per_point: int = 90,
        settle_frames: int = 30,
        mirror_horizontal: bool = False,
    ) -> None:
        self.output_path = output_path
        self.samples_per_point = samples_per_point
        self.settle_frames = settle_frames
        self.mirror_horizontal = mirror_horizontal
        self._point_index = 0
        self._settle_count = 0
        self._current_samples: list[GazeResult] = []
        self._samples: list[CalibrationSample] = []
        self.is_complete = False

    @property
    def current_point(self) -> tuple[str, float, float]:
        """Return the active calibration target."""
        return CALIBRATION_POINTS[self._point_index]

    @property
    def point_number(self) -> int:
        """Return the one-based active target number."""
        return self._point_index + 1

    @property
    def total_points(self) -> int:
        """Return the total number of calibration targets."""
        return len(CALIBRATION_POINTS)

    @property
    def progress(self) -> float:
        """Return sample collection progress for the active target."""
        return min(len(self._current_samples) / self.samples_per_point, 1.0)

    @property
    def is_settling(self) -> bool:
        """Return whether the active target is waiting before recording."""
        return self._settle_count < self.settle_frames

    def add_result(self, result: GazeResult) -> None:
        """Add one frame's gaze ratios to the active calibration target."""
        if self.is_complete or result.direction == "unknown":
            return

        if self.is_settling:
            self._settle_count += 1
            return

        self._current_samples.append(result)
        if len(self._current_samples) < self.samples_per_point:
            return

        point_name, screen_x, screen_y = self.current_point
        self._samples.append(
            CalibrationSample(
                point_name=point_name,
                screen_x=screen_x,
                screen_y=screen_y,
                eye_ratio_x=sum(self._horizontal_ratio(sample) for sample in self._current_samples)
                / len(self._current_samples),
                eye_ratio_y=sum(sample.vertical_ratio for sample in self._current_samples)
                / len(self._current_samples),
            )
        )
        self._current_samples = []
        self._settle_count = 0
        self._point_index += 1

        if self._point_index >= len(CALIBRATION_POINTS):
            self.save()
            self.is_complete = True

    def _horizontal_ratio(self, result: GazeResult) -> float:
        if self.mirror_horizontal:
            return 1.0 - result.horizontal_ratio

        return result.horizontal_ratio

    def save(self) -> None:
        """Write collected calibration samples to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for sample in self._samples:
                writer.writerow(
                    {
                        "point_name": sample.point_name,
                        "screen_x": f"{sample.screen_x:.4f}",
                        "screen_y": f"{sample.screen_y:.4f}",
                        "eye_ratio_x": f"{sample.eye_ratio_x:.4f}",
                        "eye_ratio_y": f"{sample.eye_ratio_y:.4f}",
                    }
                )


def load_calibration_profile(path: Path = CALIBRATION_FILE) -> CalibrationProfile | None:
    """Load saved calibration data and derive gaze thresholds."""
    if not path.exists():
        return None

    samples = _read_samples(path)
    by_name = {sample.point_name.lower(): sample for sample in samples}
    required = {"left", "right", "top", "bottom", "center"}
    if not required.issubset(by_name):
        return None

    center = by_name["center"]
    left = by_name["left"]
    right = by_name["right"]
    top = by_name["top"]
    bottom = by_name["bottom"]

    horizontal_low, horizontal_high = _axis_thresholds(
        center.eye_ratio_x,
        left.eye_ratio_x,
        right.eye_ratio_x,
        DEFAULT_HORIZONTAL_LOW,
        DEFAULT_HORIZONTAL_HIGH,
    )
    vertical_low, vertical_high = _axis_thresholds(
        center.eye_ratio_y,
        top.eye_ratio_y,
        bottom.eye_ratio_y,
        DEFAULT_VERTICAL_LOW,
        DEFAULT_VERTICAL_HIGH,
    )

    return CalibrationProfile(
        horizontal_low=horizontal_low,
        horizontal_high=horizontal_high,
        vertical_low=vertical_low,
        vertical_high=vertical_high,
        vertical_top=top.eye_ratio_y,
        vertical_center=center.eye_ratio_y,
        vertical_bottom=bottom.eye_ratio_y,
    )


def _read_samples(path: Path) -> list[CalibrationSample]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            CalibrationSample(
                point_name=row["point_name"],
                screen_x=float(row["screen_x"]),
                screen_y=float(row["screen_y"]),
                eye_ratio_x=float(row["eye_ratio_x"]),
                eye_ratio_y=float(row["eye_ratio_y"]),
            )
            for row in reader
        ]


def _midpoint(first: float, second: float) -> float:
    return (first + second) / 2.0


def _axis_thresholds(
    center: float,
    low_edge: float,
    high_edge: float,
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    low_threshold = (
        _midpoint(center, low_edge)
        if low_edge < center - MIN_AXIS_SEPARATION
        else center - FALLBACK_CENTER_MARGIN
    )
    high_threshold = (
        _midpoint(center, high_edge)
        if high_edge > center + MIN_AXIS_SEPARATION
        else center + FALLBACK_CENTER_MARGIN
    )

    if low_threshold >= high_threshold:
        return (default_low, default_high)

    return (_clamp(low_threshold), _clamp(high_threshold))


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))
