import sys, os, argparse
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
from palettable.matplotlib import Plasma_20 as ColorPalette
import palettable

class ModelNOCVizModule(EaselModule):
    def __init__(self):
        super().__init__()

    def init(self, InputArgs=None):
        self.Parser = argparse.ArgumentParser(description='NOCSMapModule to visualize NOCS maps and camera poses.', fromfile_prefix_chars='@')
        ArgGroup = self.Parser.add_argument_group()
        ArgGroup.add_argument('--models', nargs='+', help='Specify input OBJ model paths. * globbing is supported.', required=True)
        ArgGroup.add_argument('--normalize', help='Choose to normalize models to lie within NOCS.', action='store_true')
        self.Parser.set_defaults(normalize=False)
        ArgGroup.add_argument('--color-order', help='Choose to color models with point color order.', action='store_true')
        self.Parser.set_defaults(color_order=False)
        ArgGroup.add_argument('--half-offset', help='Offset all points by +0.5.', action='store_true')
        self.Parser.set_defaults(half_offset=False)
        ArgGroup.add_argument('--color-file', help='Choose a color from file. Should match number of points in model.', required=False)

        self.Args, _ = self.Parser.parse_known_args(InputArgs)
        if len(sys.argv) <= 1:
            self.Parser.print_help()
            exit()

        self.Models = []
        self.OBJLoaders = []

        for m in self.Args.models:
            self.Models.append(datastructures.PointSet3D())
            if self.Args.normalize == True:
                print('[ INFO ]: Normalizing models to lie within NOCS.')
                self.OBJLoaders.append(obj_loader.Loader(m, isNormalize=True))
            else:
                self.OBJLoaders.append(obj_loader.Loader(m, isNormalize=False))
            if len(self.OBJLoaders[-1].vertices) > 0:
                self.Models[-1].Points = np.array(self.OBJLoaders[-1].vertices)
            if len(self.OBJLoaders[-1].vertcolors) > 0:
                print('[ INFO ]: Found vertex colors. Will be used for rendering.')
                self.Models[-1].Colors = np.asarray(self.OBJLoaders[-1].vertcolors)
            else:
                self.Models[-1].Colors = self.Models[-1].Points

            if self.Args.color_file is not None:
                print('[ INFO ]: Coloring models with file.')
                Colors = np.load(self.Args.color_file)
                # print(self.Models[-1].Colors.shape)
                # print(Colors[:, :3].shape)
                self.Models[-1].Colors = np.asarray(Colors[:, :3]) + 0.5
            if self.Args.color_order == True:
                print('[ INFO ]: Coloring models with point order color.')
                Steps = np.linspace(0.0, 1.0, num=len(self.Models[-1]))
                Colors = ColorPalette.mpl_colormap(Steps)
                # print(self.Models[-1].Colors.shape)
                # print(Colors[:, :3].shape)
                self.Models[-1].Colors = np.asarray(Colors[:, :3])
                # # TEMP
                # np.save('colors', self.Models[-1].Points)

            if self.Args.half_offset == True:
                self.Models[-1].Points += 0.5

            self.Models[-1].update()

        self.isDrawNOCSCube = True
        self.isDrawPoints = False
        self.isDrawMesh = True
        self.isDrawBB = False
        self.isColorByOrder = False
        # self.RotateAngle = -90
        # self.RotateAxis = np.array([1, 0, 0])
        self.RotateAngle = 0
        self.RotateAxis = np.array([1, 0, 0])
        self.PointSize = 10.0
        self.nModels = len(self.Args.models)
        self.activeModelIdx = self.nModels # nModels will show all

    def step(self):
        pass

    def draw(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        Scale = 50
        Offset = -0.5
        gl.glPushMatrix()

        gl.glRotate(self.RotateAngle, self.RotateAxis[0], self.RotateAxis[1], self.RotateAxis[2])
        gl.glScale(Scale, Scale, Scale)
        gl.glTranslate(Offset, Offset, Offset)

        for Idx, m in enumerate(self.Models):
            if self.activeModelIdx != self.nModels:
                if Idx != self.activeModelIdx:
                    continue
            if self.isDrawMesh:
                self.OBJLoaders[Idx].draw(self.PointSize)
            if self.isDrawPoints:
                self.Models[Idx].draw(self.PointSize)
            if self.isDrawBB:
                self.Models[Idx].drawBB()

        if self.isDrawNOCSCube:
            drawing.drawUnitWireCube(5.0, True)
        gl.glPopMatrix()

        gl.glPopMatrix()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_T:  # Toggle models
            if self.nModels > 0:
                self.activeModelIdx = (self.activeModelIdx + 1)%(self.nModels+1)

        if a0.key() == QtCore.Qt.Key_N:
            self.isDrawNOCSCube = not self.isDrawNOCSCube
        if a0.key() == QtCore.Qt.Key_P:
            self.isDrawPoints = not self.isDrawPoints
        if a0.key() == QtCore.Qt.Key_M:
            self.isDrawMesh = not self.isDrawMesh
        if a0.key() == QtCore.Qt.Key_B:
            self.isDrawBB = not self.isDrawBB
        if a0.key() == QtCore.Qt.Key_S:
            print('[ INFO ]: Saving current model as OBJ.')
            if self.activeModelIdx == self.nModels:
                for i, m in enumerate(self.Models):
                    m.serialize('model_{}.obj'.format(str(i).zfill(3)))
            else:
                self.Models[self.activeModelIdx].serialize('model_{}.obj'.format(str(self.activeModelIdx).zfill(3)))
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
