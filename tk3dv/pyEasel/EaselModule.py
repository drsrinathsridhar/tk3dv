from abc import ABC, abstractmethod
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

# Modeled after the equivalent class in the C++ version
class EaselModule(ABC):
    def __init__(self):
        super().__init__()

    def __del__(self):
        pass

    @abstractmethod
    def init(self, argv=None):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def draw(self):
        pass

    def keyPressEvent(self, a0: QKeyEvent):
        pass
    def keyReleaseEvent(self, a0: QKeyEvent):
        pass
    def mousePressEvent(self, a0: QMouseEvent):
        pass
    def mouseReleaseEvent(self, a0: QMouseEvent):
        pass
    def mouseMoveEvent(self, a0: QMouseEvent):
        pass
    def wheelEvent(self, a0: QWheelEvent):
        pass
