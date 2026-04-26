"""Eye and iris landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None


Point = tuple[int, int]


@dataclass(frozen=True)
class EyeLandmarks:
    """Pixel coordinates for eye and iris landmarks."""

    left_eye: list[Point]
    right_eye: list[Point]
    left_iris: list[Point]
    right_iris: list[Point]


class EyeTracker:
    """Extract and draw eye landmarks from MediaPipe Face Mesh output."""

    # MediaPipe Face Mesh landmark indices with refine_landmarks=True.
    LEFT_EYE_CONTOUR = (
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    )
    RIGHT_EYE_CONTOUR = (
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    )
    LEFT_IRIS = (473, 474, 475, 476, 477)
    RIGHT_IRIS = (468, 469, 470, 471, 472)

    def __init__(self) -> None:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. Run `pip install -r requirements.txt` "
                "from the project root."
            )

    def extract(self, face_landmarks: object, frame_shape: Sequence[int]) -> EyeLandmarks:
        """Return eye and iris landmarks as pixel coordinates."""
        height, width = frame_shape[:2]

        return EyeLandmarks(
            left_eye=self._points(face_landmarks, self.LEFT_EYE_CONTOUR, width, height),
            right_eye=self._points(face_landmarks, self.RIGHT_EYE_CONTOUR, width, height),
            left_iris=self._points(face_landmarks, self.LEFT_IRIS, width, height),
            right_iris=self._points(face_landmarks, self.RIGHT_IRIS, width, height),
        )

    def draw(self, frame: object, landmarks: EyeLandmarks) -> None:
        """Draw eye outlines and iris points on a frame."""
        self._draw_eye(frame, landmarks.left_eye, color=(0, 255, 0))
        self._draw_eye(frame, landmarks.right_eye, color=(0, 255, 0))
        self._draw_points(frame, landmarks.left_iris, color=(0, 255, 255), radius=2)
        self._draw_points(frame, landmarks.right_iris, color=(0, 255, 255), radius=2)

    def _points(
        self,
        face_landmarks: object,
        indices: Sequence[int],
        width: int,
        height: int,
    ) -> list[Point]:
        points: list[Point] = []

        for index in indices:
            landmark = self._landmark_at(face_landmarks, index)
            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            points.append((x, y))

        return points

    def _landmark_at(self, face_landmarks: object, index: int) -> object:
        if hasattr(face_landmarks, "landmark"):
            return face_landmarks.landmark[index]

        return face_landmarks[index]

    def _draw_eye(self, frame: object, points: list[Point], color: tuple[int, int, int]) -> None:
        if not points:
            return

        for start, end in zip(points, points[1:] + points[:1]):
            cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

    def _draw_points(
        self,
        frame: object,
        points: list[Point],
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        for point in points:
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
