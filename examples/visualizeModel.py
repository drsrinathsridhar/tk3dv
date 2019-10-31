import sys, os
import tk3dv.pyEasel as pyEasel
from tk3dv.common import drawing
from PyQt5.QtWidgets import QApplication

from tk3dv.pyEasel import *
from EaselModule import EaselModule
from Easel import Easel
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
import OpenGL.GL as gl

from tk3dv.nocstools import datastructures, obj_loader
import PyQt5.QtCore as QtCore
import numpy as np

class ModelNOCVizModule(EaselModule):
    def __init__(self):
        super().__init__()

    def init(self, argv=None):
        # print('Using arguments: ', argv)
        self.argv = argv
        if (len(argv) is not 1+1):
            print('[ USAGE ]:', argv[0], '<ModelFileOBJ>')
            exit()

        self.ModelPoints = datastructures.PointSet3D()

        self.OBJLoader = obj_loader.Loader(argv[1], isNormalize=True)

        if len(self.OBJLoader.vertices) > 0:
            self.vertices = self.OBJLoader.vertices

        self.ModelPoints.Points = np.array(self.vertices)
        if len(self.OBJLoader.vertcolors) > 0:
            self.ModelPoints.Colors = np.asarray(self.OBJLoader.vertcolors)
        else:
            self.ModelPoints.Colors = self.ModelPoints.Points
        self.ModelPoints.update()
        self.isDrawNOCSCube = True
        self.isDrawPoints = False
        self.isDrawMesh = True
        self.RotateAngle = 0
        self.PointSize = 10.0

    def step(self):
        pass

    def draw(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        Scale = 50
        Offset = -0.5
        gl.glPushMatrix()

        gl.glRotate(self.RotateAngle, 0, 1, 0)
        gl.glScale(Scale, Scale, Scale)
        gl.glTranslate(Offset, Offset, Offset)

        if self.isDrawMesh:
            self.OBJLoader.draw(self.PointSize)
        if self.isDrawPoints:
            self.ModelPoints.draw(self.PointSize)
        if self.isDrawNOCSCube:
            drawing.drawUnitWireCube(5.0, True)
        gl.glPopMatrix()

        gl.glPopMatrix()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_N:
            self.isDrawNOCSCube = not self.isDrawNOCSCube
        if a0.key() == QtCore.Qt.Key_P:
            self.isDrawPoints = not self.isDrawPoints
        if a0.key() == QtCore.Qt.Key_M:
            self.isDrawMesh = not self.isDrawMesh
        if a0.key() == QtCore.Qt.Key_S:
            print('[ INFO ]: Saving model as OBJ.')
            self.ModelPoints.serialize('visualizer_out.obj')
            sys.stdout.flush()

        if a0.key() == QtCore.Qt.Key_Plus:  # Increase or decrese point size
            if self.PointSize < 20:
                self.PointSize = self.PointSize + 1

        if a0.key() == QtCore.Qt.Key_Minus:  # Increase or decrese point size
            if self.PointSize > 1:
                self.PointSize = self.PointSize - 1


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Easel([ModelNOCVizModule()], sys.argv)
    mainWindow.show()
    sys.exit(app.exec_())
