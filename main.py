import logging
import sys

from PySide6.QtWidgets import QApplication

from core.hub_window import HubWindow


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    window = HubWindow(camera=None)  # the shared camera is added in Phase 2
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
