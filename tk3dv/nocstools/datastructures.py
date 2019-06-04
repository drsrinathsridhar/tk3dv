import os
import sys

import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import numpy as np

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '..'))
from tk3dv.common import drawing

def backproject(DepthImage, Intrinsics, mask=None):
    # OutPoints = np.zeros([0, 3])  # Each point is a row

    # Depth image should be (DepthImage.shape) == 2 and DepthImage.dtype == 'uint16':

    # # Back project and add
    # if mask is None:
    #     DepthIdx = np.where(DepthImage > 0)
    # else:
    #     DepthIdx = np.where((mask >= 255))
    # IntrinsicsInv = np.linalg.inv(Intrinsics)
    # for i in range(0, DepthIdx[0].shape[0]):
    #     zVal = DepthImage[DepthIdx[0][i], DepthIdx[1][i]]
    #     UV = np.array([DepthIdx[1][i], DepthIdx[0][i], 1])  # Row/col to uv
    #     XYZ = np.dot(IntrinsicsInv, UV)
    #     XYZ = XYZ * (zVal / XYZ[2])
    #     # Because of differences in image coordinate systems
    #     OutPoints = np.vstack([OutPoints, np.array([-XYZ[0], -XYZ[1], XYZ[2]])])

    IntrinsicsInv = np.linalg.inv(Intrinsics)

    non_zero_mask = (DepthImage >= 0)
    idxs = np.where(non_zero_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = IntrinsicsInv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = DepthImage[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    OutPoints = pts

    return OutPoints

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

class OBJPointSet3D(PointSet3D):
    def __init__(self, OBJPath):
        if os.path.exists(OBJPath) == False:
            print('[ WARN ]: File does not exist:', OBJPath)

class NOCSMap3D(PointSet3D):
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

class DepthImage3D(PointSet3D):
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

        self.Points = backproject(DepthImage, Intrinsics, mask)
        self.Colors = np.zeros_like(self.Points)

        # print('Max depth:', np.max(self.Points[:, 2]))
        # print('Min depth:', np.min(self.Points[:, 2]))
        # print('Added', self.Points.shape, 'points.')

    def __del__(self):
        super().__del__()

class CameraIntrinsics():
    def __init__(self, matrix=None):
        self.Matrix = matrix
        self.Width = 0
        self.Height = 0

        self.PresetWidths = np.array([640, 320]) # Add more as needed
        self.PresetHeights = np.array([480, 240]) # Add more as needed

        self.DistCoeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) # Assuming upto 8 coeffs, all 0

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
        print('[ INFO ]: Using distortion coeffcients:', self.DistCoeffs)
