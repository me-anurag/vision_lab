import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.module_registry import ModuleEntry, discover_modules

log = logging.getLogger(__name__)

CARD_COLUMNS = 3


class ModuleCard(QFrame):
    """A clickable card on the hub page."""

    clicked = Signal()

    def __init__(self, title, description, icon_path=None, enabled=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(240, 170)
        self.setEnabled(enabled)
        if enabled:
            self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)

        if icon_path is not None:
            pixmap = QPixmap(str(icon_path))
            if not pixmap.isNull():
                icon = QLabel()
                icon.setPixmap(pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                icon.setAlignment(Qt.AlignCenter)
                layout.addWidget(icon)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        layout.addStretch()

    def mouseReleaseEvent(self, event):
        if self.isEnabled() and event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class HubWindow(QMainWindow):
    def __init__(self, camera=None):
        super().__init__()
        self.setWindowTitle("Vision Lab")
        self.resize(1000, 700)

        self.camera = camera
        self.current_module = None
        self.module_pages = {}  # module instance -> its page widget

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.hub_page = self._build_hub_page()
        self.stack.addWidget(self.hub_page)

    # ----- hub page -----

    def _build_hub_page(self):
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        grid.setSpacing(16)

        for i, entry in enumerate(discover_modules()):
            grid.addWidget(self._make_card(entry), i // CARD_COLUMNS, i % CARD_COLUMNS)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_widget)
        return scroll

    def _make_card(self, entry: ModuleEntry):
        if entry.cls is not None:
            try:
                module = entry.cls(self.camera)
            except Exception as exc:
                log.exception("Failed to create module %r", entry.package)
                entry.error = f"{type(exc).__name__}: {exc}"

        if entry.error is not None:
            card = ModuleCard(entry.package, "Failed to load", enabled=False)
            card.setToolTip(entry.error)
            return card

        icon_path = entry.folder / module.icon if module.icon else None
        card = ModuleCard(module.name, module.description, icon_path)
        card.clicked.connect(lambda m=module: self.open_module(m))
        return card

    # ----- module page -----

    def _build_module_page(self, module):
        page = QWidget()
        layout = QVBoxLayout(page)

        bar = QHBoxLayout()
        back = QPushButton("← Back to Hub")
        back.clicked.connect(self.back_to_hub)
        bar.addWidget(back)
        title = QLabel(module.name)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch()
        layout.addLayout(bar)

        layout.addWidget(module.widget(page), stretch=1)
        return page

    def open_module(self, module):
        if module not in self.module_pages:
            try:
                page = self._build_module_page(module)
            except Exception:
                log.exception("Failed to build UI for module %r", module.name)
                return
            self.module_pages[module] = page
            self.stack.addWidget(page)

        self.stack.setCurrentWidget(self.module_pages[module])
        self.current_module = module
        try:
            module.on_open()
        except Exception:
            log.exception("on_open failed for module %r", module.name)

    def back_to_hub(self):
        self._close_current_module()
        self.stack.setCurrentWidget(self.hub_page)

    def _close_current_module(self):
        if self.current_module is None:
            return
        try:
            self.current_module.on_close()
        except Exception:
            log.exception("on_close failed for module %r", self.current_module.name)
        self.current_module = None

    def closeEvent(self, event):
        self._close_current_module()
        super().closeEvent(event)
