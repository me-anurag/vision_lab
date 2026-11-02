from abc import ABC, abstractmethod

from PySide6.QtWidgets import QWidget


class BaseModule(ABC):
    """Base class for every Vision Lab module.

    A module lives in modules/<package>/module.py and defines one subclass of this.
    The hub creates it with the shared camera, builds its widget once, and calls
    on_open() / on_close() when the user enters or leaves the module page.
    """

    name: str = "Unnamed module"
    description: str = ""
    icon: str | None = None  # image path relative to the module folder

    def __init__(self, camera):
        self.camera = camera
        self._widget: QWidget | None = None

    def widget(self, parent) -> QWidget:
        """Return the module's widget, building it on first use."""
        if self._widget is None:
            self._widget = self.create_widget(parent)
        return self._widget

    @abstractmethod
    def create_widget(self, parent) -> QWidget:
        """Build and return the module's UI. Called once."""

    def on_open(self):
        """Called when the module page is shown: connect to frames, start the camera."""

    def on_close(self):
        """Called when leaving the module page: disconnect, stop the camera and workers."""
