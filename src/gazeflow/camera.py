"""Webcam capture helpers for GazeFlow."""

from __future__ import annotations

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

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
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

                cv2.putText(
                    frame,
                    "GazeFlow - press q or Esc to exit",
                    (20, 40),
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
            capture.release()
            cv2.destroyAllWindows()
