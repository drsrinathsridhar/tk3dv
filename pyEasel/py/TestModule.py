import sys
sys.path.append('./')
from EaselModule import EaselModule
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
import threading
from time import sleep
from common import utilities
import GLViewer as glv
import PyQt5.QtCore as QtCore

class TestModule(EaselModule, argv=None):
    def __init__(self, argv=None):
        super().__init__()

    def init(self):
        #print('One-time initialization before startup happens here.')
        pass

    def step(self):
        #print('Step.')
        pass

    def draw(self):
        #print('OpenGL drawing.')
        pass

# Some code to manage the module
class TestModuleManager(glv.GLViewer):
    def __init__(self):
        super().__init__()
        self.Module = TestModule()
        # Add more modules if needed manually
        self.init()

    def init(self):
        self.isStop = False
        self.isPause = False
        self.Mutex = threading.Lock()
        self.FPS = 0

        self.Module.init()
        # Add more modules if needed manually

        # Start step() thread
        self.StepThread = threading.Thread(target=self.start, args=(1,))
        self.StepThread.daemon = True
        self.StepThread.start()

    def start(self, Dummy):
        while (self.isStop == False):
            if (self.isPause == True):
                sleep(0.001) # Prevent CPU throttling
                continue
            self.stepAll()

    def stepAll(self):
        startTime = utilities.getCurrentEpochTime()
        self.Module.step()
        # Add more modules if needed
        endTime = utilities.getCurrentEpochTime()
        ElapsedTime = (endTime - startTime)

        if ElapsedTime < 1000:
            sleep(0.001)  # Prevent CPU throttling
            ElapsedTime += 1000

        self.FPS = 1e6 / (ElapsedTime)

    def stop(self):
        self.Mutex.acquire()
        self.isStop = not self.isStop
        self.Mutex.release()

        if self.StepThread is not None:
            self.StepThread.join()

    def togglePause(self):
        self.Mutex.acquire()
        self.isPause = not self.isPause
        self.Mutex.release()

    def moduleDraw(self):
        self.Module.draw()

    def keyPressEvent(self, a0: QKeyEvent):
        if(a0.key() == QtCore.Qt.Key_Escape):
            self.stop()

        super().keyPressEvent(a0)

    def mousePressEvent(self, a0: QMouseEvent):
        super().mousePressEvent(a0)
        # Implement class-specific functionality here

    def mouseReleaseEvent(self, a0: QMouseEvent):
        super().mouseReleaseEvent(a0)
        # Implement class-specific functionality here

    def mouseMoveEvent(self, a0: QMouseEvent):
        super().mouseMoveEvent(a0)

    def wheelEvent(self, a0: QWheelEvent):
        super().wheelEvent(a0)