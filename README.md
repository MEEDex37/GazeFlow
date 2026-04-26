# GazeFlow
Real-time webcam-based eye tracking and attention analytics using Python, OpenCV, and MediaPipe.

## Current Status

Phase 2 is implemented: the app opens the webcam, detects one face with MediaPipe Face Landmarker, and draws eye/iris landmarks on the live camera feed.

## Run

```powershell
cd "C:\VSC\Eye Tracking\GazeFlow"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Press `q` or `Esc` to close the camera window.

## Model File

Phase 2 needs the MediaPipe model at:

```text
assets/models/face_landmarker.task
```

The model is downloaded locally and ignored by git because it is a generated binary asset.
