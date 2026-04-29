# GazeFlow
Real-time webcam-based eye tracking and attention analytics using Python, OpenCV, and MediaPipe.

## Current Status

Phase 5 is implemented: the app opens the webcam, detects one face with MediaPipe Face Landmarker, draws eye/iris landmarks, displays gaze direction, detects blinks, and supports 5-point calibration.

## Run

```powershell
cd "C:\VSC\Eye Tracking\GazeFlow"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Press `c` to start calibration. Press `q` or `Esc` to close the camera window.

## Gaze Direction

The current gaze direction is estimated by comparing iris center position against the eye contour bounds. The webcam preview uses mirrored horizontal labeling so left/right matches the user's perspective. If calibration data exists, the app loads it at startup and uses it to tune the gaze thresholds.

## Calibration

Press `c` in the webcam window to start calibration. Look at each on-screen target until its progress bar fills:

```text
Top
Left
Center
Right
Bottom
```

The app saves calibration samples to:

```text
data/calibration_data.csv
```

If that file is missing, the app falls back to the default heuristic thresholds.

## Blink Detection

Blink detection is estimated from eye contour openness. The app displays the current blink count, shows `Blink Detected` when a new blink event is found, and includes a short cooldown so one blink is not counted repeatedly.

## Model File

The MediaPipe face landmark model is required at:

```text
assets/models/face_landmarker.task
```

The model is downloaded locally and ignored by git because it is a generated binary asset.
