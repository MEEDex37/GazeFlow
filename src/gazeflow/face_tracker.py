"""Face landmark detection using MediaPipe."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gazeflow-matplotlib-cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None
    mp = None
    python = None
    vision = None


class FaceTracker:
    """Detect one face and return its Face Mesh landmarks."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        if cv2 is None or mp is None or python is None or vision is None:
            raise RuntimeError(
                "MediaPipe/OpenCV dependencies are missing. Run "
                "`pip install -r requirements.txt` from the project root."
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise RuntimeError(
                "Missing MediaPipe face landmark model. Download "
                "face_landmarker.task to `assets/models/face_landmarker.task`."
            )

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: object) -> object | None:
        """Return the first detected face landmark list, or None."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
