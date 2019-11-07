import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo

class Loader(object):
    def __init__(self, path, isNormalize=False, isOverrideVertexColors=False):
        self.isVBOBound = False

        vertices = []
        normals = []
        texcoords = []
        faces = []
        vertcolors = []
        for line in open(path, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                vertices.append(tuple(map(float, values[1:4])))
                if len(values) == 7: # read vertex colors where available in values[4:7]
                    vertcolors.append(tuple(map(float, values[4:7])))
            elif values[0] == 'vn':
                normals.append(tuple(map(float, values[1:4])))
            elif values[0] == 'vt':
                texcoords.append(tuple(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    w = map(lambda x: int(x) if x else None, v.split('/'))
                    w = map(lambda x: x-1 if x != None and x > 0 else x, w)
                    face.append(tuple(w))
                faces.append(tuple(face))

        # Final data
        self.vertices = vertices
        self.normals = normals
        self.texcoords = texcoords
        self.faces = faces
        self.vertcolors = vertcolors

        # TODO: This is extremely inefficient
        self.triangle_vertices = []
        self.triangle_vertices_colors = []
        for Face in self.faces:
            self.triangle_vertices.append((self.vertices[Face[0][0]]))
            self.triangle_vertices.append((self.vertices[Face[1][0]]))
            self.triangle_vertices.append((self.vertices[Face[2][0]]))
            if len(self.vertcolors) > 0:
                self.triangle_vertices_colors.append((self.vertcolors[Face[0][0]]))
                self.triangle_vertices_colors.append((self.vertcolors[Face[1][0]]))
                self.triangle_vertices_colors.append((self.vertcolors[Face[2][0]]))

        self.vertices = self.triangle_vertices
        self.vertcolors = self.triangle_vertices_colors
        if len(self.vertcolors) > 0: # Prefer vertex colors if available
            print('[ INFO ]: Rendering using available vertex colors.')
            self.Colors = self.vertcolors

        # TODO: Do the normals need to be recomputed?
        if isNormalize is True:
            # Normalize model vertices to lie within the NOCS
            VerticesNP = np.array(self.vertices)
            # Compute extents
            XYZMin = np.min(VerticesNP, axis=0)
            XYZMax = np.max(VerticesNP, axis=0)
            DiagonalLength = np.linalg.norm(XYZMax - XYZMin)  # Get diagonal length
            self.vertices = (VerticesNP / DiagonalLength) + 0.5  # Normalize. Similar to ShapeNet normalization
            if isOverrideVertexColors or len(self.vertcolors) <= 0:
                self.Colors = self.vertices
            print('[ INFO ]: Normalization factor (diagonal length) =', DiagonalLength)

        # self.Colors = VerticesNP / DiagonalLength # Normalize. Similar to ShapeNet normalization

        print('[ INFO ]: Loaded', path, '\n\t\twith vertices/faces/normals:',
              len(self.vertices), '/', len(self.faces), '/', len(self.normals))

        self.update()

    def __del__(self):
        if self.isVBOBound:
            self.VBOPoints.delete()
            self.VBOColors.delete()

    def update(self):
        self.nPoints = len(self.vertices)
        if self.nPoints == 0:
            return

        self.nPoints = len(self.vertices)
        # Create VBO
        self.VBOPoints = glvbo.VBO(np.asarray(self.vertices))
        self.VBOColors = glvbo.VBO(np.asarray(self.Colors))
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

        if isWireFrame:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.nPoints)

        gl.glPopAttrib()


