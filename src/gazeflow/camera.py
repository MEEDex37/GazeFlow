"""Webcam capture helpers for GazeFlow."""

from __future__ import annotations

from src.gazeflow.blink_detector import BlinkDetector
from src.gazeflow.calibration import CalibrationSession, load_calibration_profile
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
        gaze_estimator = self._make_gaze_estimator()
        blink_detector = BlinkDetector()
        calibration_session: CalibrationSession | None = None
        calibration_message = ""
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
                    if calibration_session is not None:
                        calibration_session.add_result(gaze_result)
                        if calibration_session.is_complete:
                            gaze_estimator = self._make_gaze_estimator()
                            calibration_session = None
                            calibration_message = "Calibration saved to data/calibration_data.csv"

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

                if calibration_session is not None:
                    self._draw_calibration_target(frame, calibration_session)
                elif calibration_message:
                    self._draw_status(
                        frame,
                        calibration_message,
                        color=(0, 255, 255),
                        position=(20, 145),
                    )

                cv2.putText(
                    frame,
                    "GazeFlow - press c to calibrate, q or Esc to exit",
                    (20, 180),
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
                if key == ord("c") and calibration_session is None:
                    calibration_session = CalibrationSession(
                        mirror_horizontal=gaze_estimator.mirror_horizontal
                    )
                    calibration_message = ""
        finally:
            face_tracker.close()
            capture.release()
            cv2.destroyAllWindows()

    def _make_gaze_estimator(self) -> GazeEstimator:
        profile = load_calibration_profile()
        if profile is None:
            return GazeEstimator(mirror_horizontal=True)

        return GazeEstimator(
            horizontal_low=profile.horizontal_low,
            horizontal_high=profile.horizontal_high,
            vertical_low=profile.vertical_low,
            vertical_high=profile.vertical_high,
            mirror_horizontal=True,
            vertical_top=profile.vertical_top,
            vertical_center=profile.vertical_center,
            vertical_bottom=profile.vertical_bottom,
        )

    def _draw_calibration_target(self, frame: object, session: CalibrationSession) -> None:
        height, width = frame.shape[:2]
        point_name, screen_x, screen_y = session.current_point
        target = (int(width * screen_x), int(height * screen_y))
        progress_width = int(220 * session.progress)

        cv2.circle(frame, target, 26, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, target, 6, (0, 255, 255), -1, cv2.LINE_AA)
        self._draw_status(
            frame,
            (
                f"Calibration {session.point_number}/{session.total_points}: "
                f"{'hold on' if session.is_settling else 'recording'} {point_name}"
            ),
            color=(0, 255, 255),
            position=(20, 145),
        )
        cv2.rectangle(frame, (20, 155), (240, 165), (80, 80, 80), 1)
        cv2.rectangle(frame, (20, 155), (20 + progress_width, 165), (0, 255, 255), -1)

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
