"""Application entrypoint for GazeFlow."""

from src.gazeflow.camera import Camera


def main() -> None:
    """Start the live webcam preview."""
    camera = Camera()
    camera.open()


if __name__ == "__main__":
    main()
