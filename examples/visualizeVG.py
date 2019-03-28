import sys, os, argparse
from tk3dv.extern.binvox import binvox_rw
from tk3dv.common import drawing
from PyQt5.QtWidgets import QApplication
import numpy as np

from tk3dv.pyEasel import *
from EaselModule import EaselModule
from Easel import Easel
import OpenGL.GL as gl

class VGVizModule(EaselModule):
    def __init__(self):
        super().__init__()

    def init(self, argv=None):
        self.Parser = argparse.ArgumentParser(description='This module visualizes voxel grids.', fromfile_prefix_chars='@')
        ArgGroup = self.Parser.add_argument_group()
        ArgGroup.add_argument('-v', '--voxel-grid', help='Specify binvox file.', required=True)

        self.Args, _ = self.Parser.parse_known_args(argv)
        if len(sys.argv) <= 1:
            self.Parser.print_help()
            exit()

        print('[ INFO ]: Opening binvox file:', self.Args.voxel_grid)
        with open(self.Args.voxel_grid, 'rb') as f:
            self.VG = binvox_rw.read_as_3d_array(f)

        print('[ INFO ]: Opened voxel grid of dimension:', self.VG.dims)
        self.VGNZ = np.nonzero(self.VG.data)

    def step(self):
        pass

    def drawVG(self, Alpha=0.8, ScaleX=1, ScaleY=1, ScaleZ=1):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glScale(ScaleX, ScaleY, ScaleZ)

        GridSize = self.VG.dims[0]

        gl.glPushMatrix()
        gl.glScale(1 / GridSize, 1 / GridSize, 1 / GridSize)

        # TODO: Create a VoxelGrid class in nocstools and add VBO based drawing to it

        for i in range(0, len(self.VGNZ[0])):
            gl.glPushMatrix()
            gl.glTranslate(self.VGNZ[0][i], self.VGNZ[1][i], self.VGNZ[2][i])
            drawing.drawUnitWireCube(lineWidth=2.0, WireColor=(0, 0, 0))
            drawing.drawUnitCube(Alpha=Alpha, Color=(101 / 255, 67 / 255, 33 / 255))
            gl.glPopMatrix()

        gl.glPopMatrix()
        gl.glPopMatrix()

    def draw(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        gl.glTranslate(-20, -20, -20)
        self.drawVG(0.7, 40, 40, 40)

        gl.glPopMatrix()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Easel([VGVizModule()], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())