import cv2
import numpy as np
import datastructures as ds
import math
import random

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class PoseRCNNInput():
    def __init__(self, ColorImage, CoordImage, DepthImage, MaskImage, Intrinsics):
        self.ColorImage, self.CoordImage, self.DepthImage, self.MaskImage = ColorImage, CoordImage, DepthImage, MaskImage
        self.Intrinsics = Intrinsics

        # Get some stats and pre-process
        if len(self.MaskImage.shape) is not 3:
            print('[ WARN ]: Mask should be 3 channels. Please check input.')
            return

        # The red channel (3) contains mask information
        MaskChannel = self.MaskImage[:, :, 2]
        self.MaskIDs = np.unique(MaskChannel).tolist()
        if 255 in self.MaskIDs: # Background value
            self.MaskIDs.remove(255)

        print(self.MaskIDs)

        self.RGBs = []
        self.NOCImages = []
        self.DepthImages = []
        self.Masks = []

        self.NOCs = []
        self.Metrics = []

        for ID in self.MaskIDs:
            IDMask = np.uint8(MaskChannel == ID)
            self.RGBs.append(cv2.bitwise_and(self.ColorImage, self.ColorImage, mask=IDMask))
            self.NOCImages.append(cv2.bitwise_and(self.CoordImage, self.CoordImage, mask=IDMask))
            self.DepthImages.append(cv2.bitwise_and(self.DepthImage, self.DepthImage, mask=IDMask))
            self.Masks.append(IDMask * 255)

        # FOR TESTING PURPOSES ONLY
        DEBUG = False

        for Idx in range(0, len(self.MaskIDs)):
            if DEBUG:
                # Random: DEBUG
                RandRotMat = rotation_matrix(np.array([1, 0, 0]), np.random.uniform(0, 359, 1))
                RandRotMat = RandRotMat @ rotation_matrix(np.array([0, 1, 0]), np.random.uniform(0, 359, 1))
                RandRotMat = RandRotMat @ rotation_matrix(np.array([0, 0, 1]), np.random.uniform(0, 359, 1))

                RandScale = np.array([1, 1, 1]) * 17  # np.random.uniform(8, 16, 3)
                RandTrans = np.random.uniform(-70, 70, 3)
                RandOutliers = 0.1

            NOC = ds.PointSet3D()
            NOCIm = self.NOCImages[Idx]
            DepIm = self.DepthImages[Idx]
            if DEBUG == False:
                Metric = ds.DepthImage3D(DepIm, self.Intrinsics.Matrix)
            else:
                Metric = ds.PointSet3D() # DEBUG
            ColIm = self.RGBs[Idx]
            IDMask = self.Masks[Idx]
            MaskIdx = np.where((IDMask >= 255))
            RGBColors = np.zeros([0, 3])

            for i in range(0, MaskIdx[0].shape[0]):
                Val = NOCIm[MaskIdx[0][i], MaskIdx[1][i], :] / 255
                Col = ColIm[MaskIdx[0][i], MaskIdx[1][i], :] / 255
                NOC.add(1 - Val[2], Val[1], Val[0], 1 - Val[2], Val[1], Val[0]) # Flip x and z (due to OpenCV) and also left/right handed coordinate systems (due to rendernigs)
                RGBColors = np.vstack([RGBColors, np.array([Col[2], Col[1], Col[0]])])
                if DEBUG:
                    # DEBUG
                    Val = np.dot(RandRotMat, Val)
                    Val = np.multiply(Val, RandScale)
                    Val = np.add(Val, RandTrans)
                    if random.uniform(0.0, 1.0) > (1-RandOutliers): # % outliers
                        Val = Val + np.random.uniform(-500, 500 , 3)
                        # if random.uniform(0.0, 1.0) > 0.5:
                        #     Val[0] = 0
                        # if random.uniform(0.0, 1.0) > 0.5:
                        #     Val[1] = 0
                        # if random.uniform(0.0, 1.0) > 0.5:
                        #     Val[2] = 0
                    Metric.add(Val[0], Val[1], Val[2])

            Metric.Colors = RGBColors
            Metric.update()
            NOC.update()

            print('[ INFO ]: Mask', Idx, 'contains', NOC.Points.shape, 'points.')
            self.NOCs.append(NOC)
            self.Metrics.append(Metric)

    def __del__(self):
        for noc in self.NOCs:
            noc.__del__()
        for metric in self.Metrics:
            metric.__del__()

class PoseRCNNInputOverlapping(PoseRCNNInput):
    def __init__(self, ColorImage, DepthImage, DetectionData, Intrinsics):
        self.CoordImage = None
        self.MaskImage = None
        self.ColorImage, self.DepthImage, = ColorImage, DepthImage
        self.Data = DetectionData
        self.Intrinsics = Intrinsics

        self.RGBs = []
        self.NOCImages = []
        self.DepthImages = []
        self.Masks = []

        self.NOCs = []
        self.Metrics = []
        self.MaskIDs = []

        # Get some stats and pre-process
        self.AllROIs = self.Data['rois']
        self.AllMasks = self.Data['masks']
        self.AllCoords = self.Data['coords']
        self.AllClassIDs = self.Data['class_ids']

        if self.AllMasks.shape[0] != ColorImage.shape[0] or self.AllMasks.shape[1] != ColorImage.shape[1]:
            print('[ WARN ]: Image dimensions do not match. Please check input.')
            print('[ INFO ]: Mask shape:', self.AllMasks.shape)
            print('[ INFO ]: Color shape:', ColorImage.shape)
            return

        if self.AllROIs is None or self.AllMasks is None or self.AllCoords is None or self.AllClassIDs is None:
            print('[ WARN ]: Detection data is wrong. Please check input.')
            return

        self.MaskIDs = self.AllClassIDs
        if 255 in self.MaskIDs: # Background value
            self.MaskIDs.remove(255)

        print(self.MaskIDs)

        for Idx in range(0, len(self.MaskIDs)):
            IDMask = self.AllMasks[:, :, Idx]
            self.RGBs.append(cv2.bitwise_and(self.ColorImage, self.ColorImage, mask=IDMask))
            self.NOCImages.append(cv2.bitwise_and(self.AllCoords[:, :, Idx, :], self.AllCoords[:, :, Idx, :], mask=IDMask) * 255) # Scale 0-1 to 0-255
            self.DepthImages.append(cv2.bitwise_and(self.DepthImage, self.DepthImage, mask=IDMask))
            self.Masks.append(IDMask * 255)

        for Idx in range(0, len(self.MaskIDs)):
            NOC = ds.PointSet3D()
            NOCIm = self.NOCImages[Idx]
            DepIm = self.DepthImages[Idx]
            Metric = ds.DepthImage3D(DepIm, self.Intrinsics.Matrix, mask=self.Masks[Idx])
            ColIm = self.RGBs[Idx]
            IDMask = self.Masks[Idx]
            MaskIdx = np.where((IDMask >= 255))
            RGBColors = np.zeros([0, 3])

            for i in range(0, MaskIdx[0].shape[0]):
                Val = NOCIm[MaskIdx[0][i], MaskIdx[1][i], :] / 255
                Col = ColIm[MaskIdx[0][i], MaskIdx[1][i], :] / 255
                NOC.add(Val[0], Val[1], Val[2], Val[0], Val[1], Val[2])
                RGBColors = np.vstack([RGBColors, np.array([Col[2], Col[1], Col[0]])])

            Metric.Colors = RGBColors
            Metric.update()
            NOC.update()

            print('[ INFO ]: Mask', Idx, 'contains', NOC.Points.shape, 'points.')
            self.NOCs.append(NOC)
            self.Metrics.append(Metric)

    def __del__(self):
        for noc in self.NOCs:
            noc.__del__()
        for metric in self.Metrics:
            metric.__del__()