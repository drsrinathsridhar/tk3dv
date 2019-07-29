import OpenGL.GL as gl
import OpenGL.GLU as glu

from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent, QPainter
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy, QPushButton, QHBoxLayout
import PyQt5.QtCore as QtCore

import numpy as np
import math
import time

from common import drawing, utilities

# This class is modeled after the GLViewer class in Easel
# See https://github.com/drsrinathsridhar/Easel/blob/master/src/gui
class GLViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLViewer, self).__init__(parent)

        self.setWindowTitle('pyEasel')
        Fact = 4.0
        self.setGeometry(30, 30, 320 * Fact, 240 * Fact + 50)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(False)

        self.Width = self.width()
        self.Height = self.height()
        # self.setMinimumSize(640, 480)
        self.isRenderPlane = True
        self.isRenderAxis = True
        self.isRotateCamera = False
        self.isUpdateEveryStep = False
        self.RotateSpeed = 0.05
        self.RotateSpeedUpdate = 0.02
        self.isDarkMode = False

        self.SceneExtents = 1000000.0
        self.SceneHeight = self.SceneExtents / 1000.0
        self.SceneUserLimit = self.SceneExtents / 100.0
        self.LastMouseMove = QPoint(0, 0)

        self.resetCamera()

    def resetCamera(self):
        self.CameraPosition = np.array([0.0, 0.0, 100.0])
        self.LookAt = np.array([0.0, 0.0, 0.0])
        self.UpDir = np.array([0.0, 1.0, 0.0])

        self.Pitch = 0.0 # Radians
        self.Roll = 0.0
        self.Yaw = 0.0
        self.Distance = 100.0
        self.FOVY = 75.0 # Degrees

        self.Translation = np.array([0.0, 0.0, 0.0])

    def clearColor(self):
        if self.isDarkMode:
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        else:
            # gl.glClearColor(0.58, 0.58, 0.58, 1.0)
            # gl.glClearColor(0.9, 0.9, 0.9, 1.0)
            # gl.glClearColor(1, 1, 1, 1)
            gl.glClearColor(0.98, 0.98, 0.98, 1.0)

    def initializeGL(self):
        self.clearColor()

    def resizeGL(self, w: int, h: int):
        gl.glViewport(0, 0, w, h)

        self.Width = w
        self.Height = h

        self.updateState()

    def updateCamera(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        glu.gluPerspective(self.FOVY, self.Width / self.Height, 1, 50000)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        glu.gluLookAt(self.CameraPosition[0], self.CameraPosition[1], self.CameraPosition[2]
                      , self.LookAt[0], self.LookAt[1], self.LookAt[2]
                      , self.UpDir[0], self.UpDir[1], self.UpDir[2])

        gl.glMatrixMode(gl.GL_MODELVIEW)

    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def makeRotationMatrix(self):
        Rz = self.rotation_matrix(np.array([0, 0, 1]), self.Roll)
        Ry = self.rotation_matrix(Rz.dot(np.array([0, 1, 0])), self.Yaw)
        Rx = self.rotation_matrix(Ry.dot(np.array([1, 0, 0])), self.Pitch)
        RotationMatrix = Rx @ Ry @ Rz

        return RotationMatrix

    def updateState(self):
        RotMat = self.makeRotationMatrix()
        self.CameraPosition = RotMat.dot(np.array([0, 0, self.Distance])) + self.Translation
        self.LookAt = RotMat.dot(np.array([0, 0, 0])) + self.Translation
        self.UpDir = RotMat.dot(np.array([0, 1, 0]))

        self.updateCamera()

    def paintEvent(self, event):
        self.makeCurrent()

        gl.glPushMatrix()
        self.drawGL()
        gl.glPopMatrix()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.endNativePainting()
        self.drawPainter(painter)
        painter.end()

    def drawPainter(self, painter: QPainter):
        # TODO
        pass

    def moduleDraw(self):
        # Implement in module manager class (see TestModule for example)
        pass

    def drawGL(self):
        self.clearColor()
        # uint64_t Tic = Common::getCurrentEpochTime();
        self.updateState()

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # gl.glEnable(gl.GL_FOG) # Enable if fog is necessary
        # gl.glEnable(gl.GL_CULL_FACE)
        gl.glShadeModel(gl.GL_SMOOTH)

        if self.isRenderPlane:
            drawing.drawCheckerBoard(10000, 1000, 1000, self.SceneHeight)
        if self.isRenderAxis:
            drawing.drawAxes()

        self.moduleDraw()

        gl.glLoadIdentity()

        gl.glFlush()

        # for (auto & Mod: self.Modules) # Call draw for each module
        #     Mod->draw();

        # uint64_t Toc = Common::getCurrentEpochTime()
        # uint64_t ElapsedTime = Toc - Tic; # In microseconds

        # if (ElapsedTime < 1000) // Add small delay to prevent CPU from being used 100 % if the process took < 1000 us
        # {
        #     Common:: sleep(1);
        # ElapsedTime += 1000;
        # }
        #
        # if (self.RenderFPS != nullptr)
        #     (*self.RenderFPS) = 1e6 / double(ElapsedTime);

        # Disable OpenGL for Qt overlay drawing
        gl.glShadeModel(gl.GL_FLAT)
        gl.glDisable(gl.GL_DEPTH_TEST)

    def keyPressEvent(self, a0: QKeyEvent):
        if(a0.modifiers() == (QtCore.Qt.ControlModifier)):
            if(a0.key() == QtCore.Qt.Key_P):
                self.isRenderPlane = not self.isRenderPlane
                self.update()
            if (a0.key() == QtCore.Qt.Key_X):
                self.isRenderAxis = not self.isRenderAxis
                self.update()
            if (a0.key() == QtCore.Qt.Key_D):
                self.isDarkMode = not self.isDarkMode
                if self.isDarkMode:
                    print('[ INFO ]: Enabling dark mode.')
                else:
                    print('[ INFO ]: Disabling dark mode.')
                self.clearColor()
                self.update()
            if (a0.key() == QtCore.Qt.Key_R):
                self.isRotateCamera = not self.isRotateCamera
                self.isUpdateEveryStep = self.isRotateCamera
                self.update()
            if (a0.key() == QtCore.Qt.Key_Period):
                if self.RotateSpeed < (1.0 - self.RotateSpeedUpdate):
                    self.RotateSpeed += self.RotateSpeedUpdate
            if (a0.key() == QtCore.Qt.Key_Comma):
                if self.RotateSpeed > self.RotateSpeedUpdate:
                    self.RotateSpeed -= self.RotateSpeedUpdate

        if(a0.key() == QtCore.Qt.Key_Escape):
            QtCore.QCoreApplication.quit()

    def mousePressEvent(self, a0: QMouseEvent):
        self.LastMouseMove = a0.pos()

    def mouseReleaseEvent(self, a0: QMouseEvent):
        self.LastMouseMove = a0.pos()

    def mouseMoveEvent(self, a0: QMouseEvent):
        delta = 0.01
        dx = delta * (a0.x() - self.LastMouseMove.x())
        dy = delta * (a0.y() - self.LastMouseMove.y())

        if(a0.buttons() & QtCore.Qt.LeftButton):
            self.Yaw -= dx * (math.radians(30))
            self.Pitch -= dy * (math.radians(30))
        elif(a0.buttons() & QtCore.Qt.RightButton):
            RotationMatrix = self.makeRotationMatrix()
            self.Translation += RotationMatrix.dot(np.array([-dx, dy, 0])) * 300

        self.LastMouseMove = a0.pos()

        self.update()

    def wheelEvent(self, a0: QWheelEvent):
        if(a0.modifiers() == QtCore.Qt.NoModifier):
            dz = a0.angleDelta().y() * 0.01
            self.Distance *= math.pow(1.2, -dz)
            self.Distance = self.SceneUserLimit if (self.Distance > self.SceneUserLimit) else self.Distance

        if (a0.modifiers() == QtCore.Qt.ControlModifier):
            dz = a0.angleDelta().y() * 0.005
            Val = self.FOVY * math.pow(1.2, -dz)
            if Val > 1 and Val < 360:
                self.FOVY = Val

        self.update()
