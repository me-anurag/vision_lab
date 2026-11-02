from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.base_module import BaseModule


class HelloModule(BaseModule):
    """Minimal reference module. Copy this folder to start a new module."""

    name = "Hello"
    description = "Minimal example module."

    def create_widget(self, parent):
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        label = QLabel("Hello from the Hello module!")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px;")
        layout.addWidget(label)
        return widget
