from datetime import datetime
import numpy as np
import math

def getCurrentEpochTime():
    return int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds() * 1e6)

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