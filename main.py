import logging
import sys

from PySide6.QtWidgets import QApplication

from core.camera import Camera
from core.hub_window import HubWindow


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    camera = Camera()
    app.aboutToQuit.connect(camera.stop)  # make sure the webcam is released on exit
    window = HubWindow(camera=camera)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
