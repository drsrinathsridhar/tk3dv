import os, sys, json, ctypes
from tk3dv.extern import quaternions

import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import numpy as np

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '..'))
from tk3dv.common import drawing, utilities

class PointSet():
    def __init__(self):
        self.Points = None

class PointSet3D(PointSet):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.Points = np.zeros([0, 3])  # Each point is a row
        self.Colors = np.zeros([0, 3])
        self.isVBOBound = False
        self.BoundingBox = [np.zeros([3, 1]), np.zeros([3, 1])] # Bottom left and top right
        self.BBCenter = (self.BoundingBox[0] + self.BoundingBox[1]) / 2
        self.BBSize = (self.BoundingBox[1] - self.BoundingBox[0])

    def __del__(self):
        if self.isVBOBound:
            self.VBOPoints.delete()
            self.VBOColors.delete()

    def __len__(self):
        return self.Points.shape[0]

    def serialize(self, OutFile):
        with open(OutFile, 'w') as f:
            f.write("# PointSet3D serialized file\n")
            for i in range(self.nPoints):
                f.write('v {:.4f} {:.4f} {:.4f}\n'.format(self.Points[i, 0], self.Points[i, 1], self.Points[i, 2]))

    def updateBoundingBox(self):
        self.BoundingBox[0] = np.min(self.Points, axis=0)
        self.BoundingBox[1] = np.max(self.Points, axis=0)

        self.BBCenter = (self.BoundingBox[0] + self.BoundingBox[1]) / 2
        self.BBSize = (self.BoundingBox[1] - self.BoundingBox[0])

    def update(self):
        if self.Points.shape[0] == 0 or self.Colors.shape[0] == 0:
            return

        # Create VBO
        self.nPoints = len(self.Points)
        self.VBOPoints = glvbo.VBO(self.Points)
        self.VBOColors = glvbo.VBO(self.Colors)
        self.isVBOBound = True

        self.updateBoundingBox()

    def addAll(self, Points, Colors=None):
        self.Points = Points.astype(np.float)
        MaxVal = np.max(self.Points)
        if np.all(Colors) == None:
            if MaxVal <= 1.0:
                MaxVal = 1.0
            self.Colors = Points / MaxVal
        else:
            self.Colors = Colors

    def appendAll(self, Points, Colors=None):
        NewPoints = Points.astype(np.float)
        self.Points = np.vstack((self.Points, NewPoints))
        MaxVal = np.max(NewPoints)
        if np.all(Colors) == None:
            if MaxVal <= 1.0:
                MaxVal = 1.0
            Colors = Points / MaxVal

        self.Colors = np.vstack((self.Colors, Colors))

    def add(self, x, y, z, r = 0, g = 0, b = 0):
        self.Points = np.vstack([self.Points, np.array([x, y ,z])])
        self.Colors = np.vstack([self.Colors, np.array([r, g, b])])

    def drawBB(self, LineWidth = 1):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        gl.glTranslate(self.BBCenter[0], self.BBCenter[1], self.BBCenter[2])
        gl.glScale(self.BBSize[0], self.BBSize[1], self.BBSize[2])
        gl.glTranslate(-0.5, -0.5, -0.5) # Move box origin to center
        drawing.drawUnitWireCube(LineWidth, False)

        gl.glPopMatrix()

    def draw(self, pointSize = 10):
        if self.isVBOBound == False:
            print('[ WARN ]: VBOs not bound. Call update().')
            return

        gl.glPushAttrib(gl.GL_POINT_BIT)
        gl.glPointSize(pointSize)

        self.VBOPoints.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOPoints)

        self.VBOColors.bind()
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glColorPointer(3, gl.GL_DOUBLE, 0, self.VBOColors)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nPoints)

        gl.glPopAttrib()

class NOCSMap(PointSet3D):
    def __init__(self, NOCSMap, RGB=None, Color=None):
        super().__init__()
        self.createNOCSFromNM(NOCSMap, RGB, Color)

    def createNOCSFromNM(self, NOCSMap, RGB=None, Color=None):
        ValidIdx = np.where(np.all(NOCSMap != [255, 255, 255], axis=-1))
        ValidPoints = NOCSMap[ValidIdx[0], ValidIdx[1]] / 255

        # The PC can be colored with: (1) NOCS color, (2) RGB color, (3) Uniform color
        NOCSColors = ValidPoints
        RGBColors = None
        if RGB is not None:
            RGBColors = RGB[ValidIdx[0], ValidIdx[1]] / 255
        UniColors = None
        if Color is not None:
            ColorNP = np.asarray(Color) / 255
            UniColors = np.transpose(np.repeat(ColorNP[:, np.newaxis], NOCSMap[ValidIdx].shape[0], axis=1))

        self.addAll(ValidPoints, Colors=RGBColors)

    def __del__(self):
        super().__del__()

class VoxelGrid(PointSet3D):
    def __init__(self, BinVoxGrid):
        super().__init__()
        self.VG = BinVoxGrid
        self.GridSize = self.VG.dims[0]
        self.VGNZ = np.nonzero(self.VG.data)
        # We are treating VoxelGrid as a point cloud with unit cube sie limits
        # All 'on' voxels are a point in the point cloud. The center of a voxel is the position of the point
        self.DefaultColor = (101 / 255, 67 / 255, 33 / 255)
        self.VGCorners = np.zeros([0, 3], dtype=np.float)  # Each point is a row
        self.VGColors = np.zeros([0, 3], dtype=np.float)  # Each point is a row
        self.VGIndices = np.zeros([0,1], dtype=np.uint32)  # Each element is an index
        self.VGVBO = []
        self.nVGCorners = 0
        self.isVBOBound = False

        self.createVG()

    def update(self):
        # super().update()
        self.createVGVBO()

    def createVGVBO(self):
        self.nVGCorners = int(len(self.VGCorners) / 3)
        self.VBOPoints = glvbo.VBO(self.VGCorners)
        self.VBOColors = glvbo.VBO(self.VGColors)
        self.VBOIndices = glvbo.VBO(self.VGIndices, target=gl.GL_ELEMENT_ARRAY_BUFFER)

        # self.VGVBO = gl.glGenBuffers(3)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VGVBO[0])
        # gl.glBufferData(gl.GL_ARRAY_BUFFER,
        #                 len(self.VGCorners) * 4,  # byte size
        #                 (ctypes.c_float * len(self.VGCorners))(*self.VGCorners),
        #                 gl.GL_STATIC_DRAW)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VGVBO[1])
        # gl.glBufferData(gl.GL_ARRAY_BUFFER,
        #                 len(self.VGColors) * 4,  # byte size
        #                 (ctypes.c_float * len(self.VGColors))(*self.VGColors),
        #                 gl.GL_STATIC_DRAW)
        # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.VGVBO[2])
        # gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
        #                 len(self.VGIndices) * 4,  # byte size
        #                 (ctypes.c_uint * len(self.VGIndices))(*self.VGIndices),
        #                 gl.GL_STATIC_DRAW)

        self.isVBOBound = True

    def createVG(self, Color=None):
        # # TESTING
        #
        # self.VGCorners = np.vstack([self.VGCorners, np.asarray([0, 0, 0])])
        # self.VGCorners = np.vstack([self.VGCorners, np.asarray([1, 0, 0])])
        # self.VGCorners = np.vstack([self.VGCorners, np.asarray([1, 1, 0])])
        # self.VGCorners = np.vstack([self.VGCorners, np.asarray([0, 0, 0])])
        # self.VGCorners = np.vstack([self.VGCorners, np.asarray([0, 1, 0])])
        # self.VGCorners = np.vstack([self.VGCorners, np.asarray([1, 1, 0])])
        #
        # self.VGIndices = np.vstack([self.VGIndices, np.asarray([0, 1, 2, 3, 4, 5]).reshape((-1, 1))])
        #
        # self.VGColors = self.VGCorners
        #
        # print(self.VGCorners)
        # print(self.VGIndices)

        for i in range(0, len(self.VGNZ[0])):
            VoxelCenter = (np.array([self.VGNZ[0][i], self.VGNZ[1][i], self.VGNZ[2][i]]) + 0.5) / self.GridSize
            self.add(VoxelCenter[0], VoxelCenter[1], VoxelCenter[2], VoxelCenter[0], VoxelCenter[1], VoxelCenter[2])

            # Create vertices of voxels
            VO = (np.array([self.VGNZ[0][i], self.VGNZ[1][i], self.VGNZ[2][i]])) / self.GridSize # Voxel origin
            VS = 1 / self.GridSize # Voxel side
            Corners = [
                        VO[0], VO[1], VO[2],
                        VO[0] + VS, VO[1], VO[2],
                        VO[0] + VS, VO[1] + VS, VO[2],
                        VO[0], VO[1] + VS, VO[2],
                        VO[0], VO[1] + VS, VO[2] + VS,
                        VO[0] + VS, VO[1] + VS, VO[2] + VS,
                        VO[0] + VS, VO[1], VO[2] + VS,
                        VO[0], VO[1], VO[2] + VS,
                    ]
            self.VGCorners = np.vstack([self.VGCorners, np.asarray(Corners).reshape((-1, 3))])

            SI = i * 8 # start index
            Indices = [
                    SI+0, SI+1, SI+2, SI+2, SI+3, SI+0,
                    SI+0, SI+3, SI+4, SI+4, SI+7, SI+0,
                    SI+4, SI+7, SI+6, SI+6, SI+5, SI+4,
                    SI+0, SI+7, SI+6, SI+6, SI+1, SI+0,
                    SI+1, SI+6, SI+5, SI+5, SI+2, SI+1,
                    SI+3, SI+4, SI+5, SI+5, SI+2, SI+3,
                    ]
            self.VGIndices = np.vstack([self.VGIndices, np.asarray(Indices).reshape((-1, 1))])

            if Color is None:
                Color = self.DefaultColor
            for kk in range(0, 8):
                self.VGColors = np.vstack([self.VGColors, np.asarray(Color).reshape((-1, 3))])

        # print(len(self.VGNZ[0]))
        # print(self.VGCorners.shape)
        # print(self.VGIndices.shape)
        self.update()

    def drawVG(self, Alpha=0.8, ScaleX=1, ScaleY=1, ScaleZ=1):
        if self.isVBOBound == False:
            print('[ WARN ]: Voxel grid VBOs not bound.')

        gl.glPushAttrib(gl.GL_POINT_BIT)
        gl.glPointSize(20)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glScale(ScaleX, ScaleY, ScaleZ)

        if self.VBOPoints is not  None:
            self.VBOPoints.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_DOUBLE, 0, self.VBOPoints)

        if self.VBOColors is not  None:
            self.VBOColors.bind()
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(3, gl.GL_DOUBLE, 0, self.VBOColors)

        if self.VBOIndices is not  None:
            # gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.nVGCorners)
            self.VBOIndices.bind()
        gl.glDrawElements(gl.GL_TRIANGLES, int(len(self.VGIndices)), gl.GL_UNSIGNED_INT, None)

        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        gl.glPopAttrib()


        # gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        # gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VGVBO[0])
        # gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VGVBO[1])
        # gl.glColorPointer(3, gl.GL_FLOAT, 0, None)
        # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.VGVBO[2])
        # gl.glDrawElements(gl.GL_LINE_STRIP, len(self.VGIndices), gl.GL_UNSIGNED_INT, None)
        # gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        # gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        #
        # GridSize = self.VG.dims[0]
        #
        # gl.glPushMatrix()
        # gl.glScale(1 / GridSize, 1 / GridSize, 1 / GridSize)
        #
        # # TODO: Create a VoxelGrid class in nocstools and add VBO based drawing to it
        # # Add VBO-based drawing for cubes, gorund plane, etc.
        #
        # for i in range(0, len(self.VGNZ[0])):
        #     gl.glPushMatrix()
        #     gl.glTranslate(self.VGNZ[0][i], self.VGNZ[1][i], self.VGNZ[2][i])
        #     drawing.drawUnitWireCube(lineWidth=2.0, WireColor=(0, 0, 0))
        #     drawing.drawUnitCube(Alpha=Alpha, Color=self.DefaultColor)
        #     gl.glPopMatrix()
        #
        # gl.glPopMatrix()
        #
        gl.glPopMatrix()


class DepthImage(PointSet3D):
    def __init__(self, DepthImage, Intrinsics, mask=None):
        super().__init__()
        self.createFromDepthImage(DepthImage, Intrinsics, mask)

    def createFromDepthImage(self, DepthImage, Intrinsics, mask=None):
        self.Intrinsics = Intrinsics
        if len(DepthImage.shape) == 3:
            # This is encoded depth image, let's convert
            Depth16 = np.uint16(DepthImage[:, :, 1]*256) + np.uint16(DepthImage[:, :, 2]) # NOTE: RGB is actually BGR in opencv
            Depth16 = Depth16.astype(np.uint16)
            self.DepthImage16 = Depth16
        elif len(DepthImage.shape) == 2 and DepthImage.dtype == 'uint16':
            self.DepthImage16 = DepthImage
        else:
            print('[ WARN ]: Unsupported depth type.')
            return

        self.Points = utilities.backproject(DepthImage, Intrinsics, mask)
        self.Colors = np.zeros_like(self.Points)

        # print('Max depth:', np.max(self.Points[:, 2]))
        # print('Min depth:', np.min(self.Points[:, 2]))
        # print('Added', self.Points.shape, 'points.')

    def __del__(self):
        super().__del__()

class CameraIntrinsics():
    def __init__(self, matrix=None, fromFile=None):
        self.Matrix = matrix
        self.Width = 0
        self.Height = 0

        self.PresetWidths = np.array([640, 320]) # Add more as needed
        self.PresetHeights = np.array([480, 240]) # Add more as needed

        self.DistCoeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) # Assuming upto 8 coeffs, all 0

        if fromFile is not None:
            self.init_with_file(fromFile)

    def init_with_file(self, FileName):
        with open(FileName) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.Matrix = np.identity(3, np.float32)

        ## SAMPLE
        # # fx, fy, cx, cy[, w, h[, k1, k2, p1, p2, k3, k4, k5, k6]]
        # 571, 571, 319.5, 239.5, 640, 480, 0, 0, 0, 0, 0, 0, 0, 0

        for line in content:
            if line[0] == '#':
                continue
            Params = [x.strip() for x in line.split(',')]
            nParams = len(Params)

            if nParams != 4 and nParams != 6 and nParams != 14:
                raise RuntimeError('[ ERR ]: Unsupported number of input parameters {}.'.format(nParams))

            self.Matrix[0, 0] = Params[0] # fx
            self.Matrix[1, 1] = Params[1] # fy
            self.Matrix[0, 2] = Params[2] # cx
            self.Matrix[1, 2] = Params[3] # cy

            # Width, Height
            if nParams == 4:
                self.Width = self.PresetWidths[np.argmin(np.abs(self.PresetWidths - float(Params[2])*2))]
                self.Height = self.PresetHeights[np.argmin(np.abs(self.PresetHeights - float(Params[3])*2))]

                print('[ WARN ]: No image height and width passed. Finding the closest standard size based on the principal point.')
            elif nParams == 6:
                self.Width = int(Params[4])
                self.Height = int(Params[5])
            elif nParams == 14:
                self.Width = int(Params[4])
                self.Height = int(Params[5])

                for i in range(0, 8):
                    self.DistCoeffs[i] = float(Params[6+i])


        print('[ INFO ]: Using intrinsics:\n', self.Matrix)
        print('[ INFO ]: Using width, height - {}, {}.'.format(self.Width, self.Height))
        print('[ INFO ]: Using distortion coefficients:', self.DistCoeffs)

class CameraExtrinsics():
    def __init__(self, rotation=np.identity(3), translation=np.array([0, 0, 0]), fromFile=None):
        self.Rotation = rotation
        self.Translation = translation

        if fromFile is not None:
            self.deserialize(fromFile)

    def serialize(self, OutFile):
        print('[ WARN ]: CameraExtrinsics.serialize() not yet implemented.')
        pass

    def deserialize(self, InJSONFile):
        with open(InJSONFile) as f:
            data = json.load(f)
            # Loading convention: Flip sign of x position, flip signs of quaternion z, w
            P = np.array([data['position']['x'], data['position']['y'], data['position']['z']])
            Quat = np.array([data['rotation']['w'], data['rotation']['x'], data['rotation']['y'],
                             data['rotation']['z']])  # NOTE: order is w, x, y, z
            # Cajole transforms to work
            P[0] *= -1
            # P += 0.5 # Hack to offset to NOCS center
            Quat = np.array([Quat[0], Quat[1], -Quat[2], -Quat[3]])

            self.Translation = P
            R = quaternions.quat2mat(Quat).T
            self.Rotation = R

class Camera():
    def __init__(self, Extrinsics=CameraExtrinsics(), Intrinsics=CameraIntrinsics()):
        self.Extrinsics = Extrinsics
        self.Intrinsics = Intrinsics

    def draw(self, Color=None, isF = False, Length=5.0):
        gl.glPushMatrix()

        ScaleRotMat = np.identity(4)
        R, C = self.Extrinsics.Rotation, self.Extrinsics.Translation
        ScaleRotMat[:3, :3] = R

        gl.glTranslate(C[0], C[1], C[2])
        gl.glMultMatrixf(ScaleRotMat)
        if isF:
            gl.glRotate(180, 1, 0, 0)

        drawing.drawAxes(Length, Color=Color)
        gl.glPopMatrix()



