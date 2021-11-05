import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo

class TrimeshVisualizer(object):
    def __init__(self, TrimeshObject):
        self.isVBOBound = False

        self.Trimesh = TrimeshObject
        self.Vertices = self.Trimesh.vertices
        self.Normals = self.Trimesh.face_normals
        self.Faces = self.Trimesh.faces
        self.VertColors = self.Trimesh.visual.vertex_colors

        self.update()

    def __del__(self):
        if self.isVBOBound:
            self.VBOPoints.delete()
            self.VBOColors.delete()

    def update(self):
        self.nPoints = len(self.Vertices)
        if self.nPoints == 0:
            return

        self.nPoints = len(self.Vertices)
        # Create VBO
        self.VBOPoints = glvbo.VBO(np.asarray(self.Vertices))
        self.VBOColors = glvbo.VBO(np.asarray(self.VertColors))
        self.isVBOBound = True

    def draw(self, PointSize=10.0, isWireFrame=False):
        if self.isVBOBound == False:
            print('[ WARN ]: VBOs not bound. Call update().')
            return

        gl.glPushAttrib(gl.GL_POINT_BIT)
        gl.glPointSize(PointSize)

        if self.VBOPoints is not None:
            self.VBOPoints.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOPoints)

        if self.VBOColors is not None:
            self.VBOColors.bind()
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(3, gl.GL_DOUBLE, 0, self.VBOColors)

        if len(self.Faces) > 0:
            if isWireFrame:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.nPoints)
        else:
            gl.glDrawArrays(gl.GL_POINTS, 0, self.nPoints)

        if self.VBOColors is not None:
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        if self.VBOPoints is not None:
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        gl.glPopAttrib()


