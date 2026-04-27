"""Webcam capture helpers for GazeFlow."""

from __future__ import annotations

from src.gazeflow.blink_detector import BlinkDetector
from src.gazeflow.eye_tracker import EyeTracker
from src.gazeflow.face_tracker import FaceTracker
from src.gazeflow.gaze_estimator import GazeEstimator

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None


class Camera:
    """Open and display frames from a webcam."""

    def __init__(self, camera_index: int = 0, window_name: str = "GazeFlow") -> None:
        self.camera_index = camera_index
        self.window_name = window_name

    def open(self) -> None:
        """Open the webcam feed and display frames until the user exits."""
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. Run `pip install -r requirements.txt` "
                "from the project root, then try again."
            )

        face_tracker = FaceTracker()
        eye_tracker = EyeTracker()
        gaze_estimator = GazeEstimator(mirror_horizontal=True)
        blink_detector = BlinkDetector()
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            face_tracker.close()
            raise RuntimeError(
                f"Could not open webcam at camera index {self.camera_index}. "
                "Check camera permissions or try another camera index."
            )

        print("GazeFlow webcam preview started. Press 'q' or Esc to exit.")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                face_landmarks = face_tracker.detect(frame)
                if face_landmarks is None:
                    self._draw_status(frame, "No face detected", color=(0, 0, 255))
                else:
                    eye_landmarks = eye_tracker.extract(face_landmarks, frame.shape)
                    eye_tracker.draw(frame, eye_landmarks)
                    gaze_result = gaze_estimator.estimate(eye_landmarks)
                    blink_result = blink_detector.detect(eye_landmarks)
                    self._draw_status(frame, "Face and eyes detected", color=(0, 255, 0))
                    self._draw_status(
                        frame,
                        (
                            f"{gaze_result.label} "
                            f"(x={gaze_result.horizontal_ratio:.2f}, "
                            f"y={gaze_result.vertical_ratio:.2f})"
                        ),
                        color=(255, 255, 0),
                        position=(20, 75),
                    )
                    blink_text = (
                        "Blink Detected"
                        if blink_result.blink_detected
                        else f"Blinks: {blink_result.blink_count}"
                    )
                    self._draw_status(
                        frame,
                        f"{blink_text} (open={blink_result.openness_ratio:.2f})",
                        color=(0, 165, 255),
                        position=(20, 110),
                    )

                cv2.putText(
                    frame,
                    "GazeFlow - press q or Esc to exit",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            face_tracker.close()
            capture.release()
            cv2.destroyAllWindows()

    def _draw_status(
        self,
        frame: object,
        text: str,
        color: tuple[int, int, int],
        position: tuple[int, int] = (20, 40),
    ) -> None:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
