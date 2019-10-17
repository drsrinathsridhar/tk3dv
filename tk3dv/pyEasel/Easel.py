import threading
from time import sleep

import GLViewer as glv
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

from tk3dv.common import utilities
import math

# Some code to manage the module
class Easel(glv.GLViewer):
    def __init__(self, OtherModules=[], argv=None):
        super().__init__()
        self.setWindowTitle('pyEasel')
        self.isRenderPlane = False
        self.isRenderAxis = True

        self.Modules = []
        self.Modules.extend(OtherModules)
        self.argv = argv

        self.init()

    def init(self):
        self.isStop = False
        self.isPause = False
        self.Mutex = threading.Lock()
        self.FPS = 0

        print('[ INFO ]: Initializing all modules.')
        for Mod in self.Modules:
            Mod.init(self.argv)

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

        for Mod in self.Modules:
            Mod.step()

        endTime = utilities.getCurrentEpochTime()
        ElapsedTime = (endTime - startTime)

        if ElapsedTime < 1000:
            sleep(0.001)  # Prevent CPU throttling
            ElapsedTime += 1000

        self.FPS = 1e6 / (ElapsedTime)

        if self.isRotateCameraStack[self.activeCamStackIdx]:
            self.YawStack[self.activeCamStackIdx] += math.radians(self.RotateSpeedStack[self.activeCamStackIdx])
        if self.isUpdateEveryStep:
            self.update() # A bit hacky to force draw call after step

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
        for Mod in self.Modules:
            Mod.draw()

    def keyPressEvent(self, a0: QKeyEvent):
        if(a0.key() == QtCore.Qt.Key_Escape):
            self.stop()
            for Mod in self.Modules:
                Mod.__del__()

        super().keyPressEvent(a0)
        for Mod in self.Modules:
            Mod.keyPressEvent(a0)
        self.update()

    def mousePressEvent(self, a0: QMouseEvent):
        super().mousePressEvent(a0)
        for Mod in self.Modules:
            Mod.mousePressEvent(a0)
        self.update()

    def mouseReleaseEvent(self, a0: QMouseEvent):
        super().mouseReleaseEvent(a0)
        for Mod in self.Modules:
            Mod.mouseReleaseEvent(a0)
        self.update()

    def mouseMoveEvent(self, a0: QMouseEvent):
        super().mouseMoveEvent(a0)
        for Mod in self.Modules:
            Mod.mouseMoveEvent(a0)
        self.update()

    def wheelEvent(self, a0: QWheelEvent):
        super().wheelEvent(a0)
        for Mod in self.Modules:
            Mod.wheelEvent(a0)
        self.update()
