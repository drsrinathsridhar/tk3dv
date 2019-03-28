import sys, os, argparse
sys.path.append(os.path.dirname(__file__) + './extern/')
from tk3dv.extern.binvox import binvox_rw
from tk3dv.common import drawing
from PyQt5.QtWidgets import QApplication

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

        print(self.VG.dims)

    def step(self):
        pass

    def drawVG(self, Alpha=0.8, ScaleX=1, ScaleY=1, ScaleZ=1):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glScale(ScaleX, ScaleY, ScaleZ)

        drawing.drawUnitWireCube(lineWidth=5.0, WireColor=(0, 0, 0))
        drawing.drawUnitCube(isRainbow=True, Alpha=Alpha)

        gl.glPopMatrix()

    def draw(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        gl.glTranslate(-20, -20, -20)
        self.drawVG(0.5, 40, 40, 40)

        gl.glPopMatrix()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Easel([VGVizModule()], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())