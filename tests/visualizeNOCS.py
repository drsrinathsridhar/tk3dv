import sys, os
sys.path.append(os.path.dirname(__file__) + './extern/')
import tk3dv.pyEasel as pyEasel
from PyQt5.QtWidgets import QApplication

from tk3dv.pyEasel import *
from EaselModule import EaselModule
from Easel import Easel
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
import OpenGL.GL as gl

class NOCVizModule(EaselModule):
    def __init__(self):
        super().__init__()

    def init(self, argv=None):
        print('Using arguments: ', argv)

    def step(self):
        pass

    def drawNOCS(self, Alpha=0.8, ScaleX=1, ScaleY=1, ScaleZ=1):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glScale(ScaleX, ScaleY, ScaleZ)
        # Drawing CCW
        gl.glBegin(gl.GL_QUADS)

        for i in range(0, 2):
            # Bottom and Top
            gl.glColor4f( 0.0, 0.0, i, Alpha)
            gl.glVertex3f(0.0, 0.0, i)
            gl.glColor4f( 1.0, 0.0, i, Alpha)
            gl.glVertex3f(1.0, 0.0, i)
            gl.glColor4f( 1.0, 1.0, i, Alpha)
            gl.glVertex3f(1.0, 1.0, i)
            gl.glColor4f( 0.0, 1.0, i, Alpha)
            gl.glVertex3f(0.0, 1.0, i)

            # Right and Left
            gl.glColor4f( i, 0.0, 0.0, Alpha)
            gl.glVertex3f(i, 0.0, 0.0)
            gl.glColor4f( i, 1.0, 0.0, Alpha)
            gl.glVertex3f(i, 1.0, 0.0)
            gl.glColor4f( i, 1.0, 1.0, Alpha)
            gl.glVertex3f(i, 1.0, 1.0)
            gl.glColor4f( i, 0.0, 1.0, Alpha)
            gl.glVertex3f(i, 0.0, 1.0)

            # Front and Back
            gl.glColor4f( 0.0, i, 0.0, Alpha)
            gl.glVertex3f(0.0, i, 0.0)
            gl.glColor4f( 1.0, i, 0.0, Alpha)
            gl.glVertex3f(1.0, i, 0.0)
            gl.glColor4f( 1.0, i, 1.0, Alpha)
            gl.glVertex3f(1.0, i, 1.0)
            gl.glColor4f( 0.0, i, 1.0, Alpha)
            gl.glVertex3f(0.0, i, 1.0)

        gl.glEnd()
        gl.glPopMatrix()

    def draw(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        gl.glTranslate(-20, -20, -20)
        # Draw other NOCS here
        self.drawNOCS(0.5, 40, 40, 40)

        gl.glPopMatrix()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Easel([NOCVizModule()], sys.argv)
    mainWindow.show()
    sys.exit(app.exec_())