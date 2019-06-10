'''
Created on 01.11.2012

@author: David
mean reprojection error: 0.551599651484px
root mean squared reprojection error: 0.615873177017px
'''
import numpy, math, scipy, cv2
from scipy import optimize

# computes the 2d distance between vector a and b
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


_EPS = numpy.finfo(float).eps * 4.0


# Gold Standard Algorithm for estimating P (Multiple View Geometry Sec. Edition, page:181, (7.1))
# returns (p, camera center, calibration matrix, rotation matrix)
def calculateCameraParameters(correspondences):
    normCorr, t, u = normalize(correspondences)
    p = dlt(normCorr)
    p = nonLinearOptimization(p, normCorr)
    p = denormalize(p, t, u)
    c, k, r, Flip = extractCameraParameters(p)

    return p, c, k, r, Flip

# construct the matrix A for the estimation of P (Multiple View Geometry Sec. Edition, page:179, (7.2))
def constructMatrixA(correspondences):
    matrixList = []
    for (x, bigX) in correspondences:
        matrixList.append([0, 0, 0, 0, -bigX[0], -bigX[1], -bigX[2], -1, x[1] * bigX[0], x[1] * bigX[1], x[1] * bigX[2], x[1] * 1])
        matrixList.append([bigX[0], bigX[1], bigX[2], 1, 0, 0, 0, 0, -x[0] * bigX[0], -x[0] * bigX[1], -x[0] * bigX[2], -x[0] * 1])
    return numpy.array(matrixList)


# direct linear transform of the correspondences
def dlt(correspondences):
    matrix = constructMatrixA(correspondences)
    _, _, v = numpy.linalg.svd(matrix)
    # take the last column
    pcolumn = v[-1, : ]
    p = numpy.reshape(pcolumn, (3, 4))
    # print("dlt-estimation of P:")
    # print(p)
    return p

# normalize the correspondences (mean origin and mean length)
# returns the normalized correspondences and the transformation matrices (t,u)
def normalize(correspondences):
    # transform correspondences from tuples into numpy.array's
    corr = []
    for (imageCoord, worldCoord) in correspondences:
        corr.append((numpy.array(imageCoord), numpy.array(worldCoord)))
        
    # compute mean origin
    imageOrigin = numpy.array([0.0, 0.0])
    worldOrigin = numpy.array([0.0, 0.0, 0.0])
    for (imageCoord, worldCoord) in corr:
        imageOrigin += imageCoord
        worldOrigin += worldCoord
    imageOrigin /= len(correspondences)
    worldOrigin /= len(correspondences)
    
    # compute mean norm
    imageNorm = 0.0
    worldNorm = 0.0
    for (imageCoord, worldCoord) in corr:
        imageNorm += numpy.linalg.norm(imageOrigin - imageCoord)
        worldNorm += numpy.linalg.norm(worldOrigin - worldCoord)
    imageNorm /= len(correspondences)
    worldNorm /= len(correspondences)
    tscale = math.sqrt(2) / imageNorm
    uscale = math.sqrt(3) / worldNorm
    
    # create normalized correspondences by multiplying with matrix t, u
    t = numpy.array([[tscale, 0, -imageOrigin[0] * tscale ], [0, tscale, -imageOrigin[1] * tscale ], [0, 0, 1]])
    u = numpy.array([[uscale, 0, 0, -worldOrigin[0] * uscale ], [0, uscale, 0, -worldOrigin[1] * uscale ], [0, 0, uscale, -worldOrigin[2] * uscale], [0, 0, 0, 1]])
    normalizedCorrespondences = []
    for (imageCoord, worldCoord) in corr:
        normalizedImageCoord = numpy.dot(t, numpy.append(imageCoord, [1]))
        normalizedWorldCoord = numpy.dot(u, numpy.append(worldCoord, [1]))
        normalizedCorrespondences.append((normalizedImageCoord, normalizedWorldCoord))
    return normalizedCorrespondences, t, u

def denormalize(p, t, u):
    return numpy.dot(numpy.dot(numpy.linalg.inv(t), p), u)


# extract the camera paramters from p 
# returns (camera center, calibration matrix, rotation matrix)
def extractCameraParameters(p):
    # k, r, c, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    #
    # return c, k, r, False

    # Own implementation, seemingly slightly different from OpenCV's
    # calculate camera centre c = -A^(-1)*b, P = [A|b]
    a = p[0:3, 0:3]
    b = p[:, -1]
    c = -numpy.dot(numpy.linalg.inv(a), b)
    # rq decomposition
    k, r = scipy.linalg.rq(a)

    Flip = False
    # Ensure estimated K is an appropriate format
    k = k / k[-1, -1]  # Should be 1
    if k[0, 0] < 0 and k[1, 1] < 0:
        k[0, 0] = -k[0, 0]
        k[1, 1] = -k[1, 1]
        Flip = False
    if k[0, 0] < 0 and k[1, 1] >= 0:
        k[0, 0] = -k[0, 0]
        Flip = True
    if k[0, 0] >= 0 and k[1, 1] < 0:
        k[1, 1] = -k[1, 1]
        Flip = True

    return c, k, r, Flip


def reprojectionError(p, correspondences):
    projectionerr = []
    p = numpy.reshape(p[0:12], (3, 4))
    for (x2d, x3d) in correspondences:
        projectedPos = numpy.dot(p , x3d)
        projectedPos /= projectedPos[2]
        dst = distance((projectedPos[0], projectedPos[1]), x2d)
        projectionerr.append(dst)
    return numpy.asarray(projectionerr)


def nonLinearOptimization(p, correspondences):
    # convert p matrix into vector, p.flatten() somehow doesn't work :(
    pflat = numpy.array([p[0, 0], p[0, 1], p[0, 2], p[0, 3], p[1, 0], p[1, 1], p[1, 2], p[1, 3], p[2, 0], p[2, 1], p[2, 2], p[2, 3]])
    p = optimize.leastsq(reprojectionError, pflat, correspondences, factor=0.01)
    p = numpy.reshape(p[0], (3, 4))
    # print("optimized p")
    # print(p)
    return p

# from Quaternion import Quat
# import Quaternion

# def quaternion_to_matrix(quatArr):
#     quatArr = Quaternion.normalize(quatArr)
#     q = Quat(attitude=quatArr)
#     return q.transform
#
#
# def quaternion_from_matrix(matrix):
#     q = Quat(attitude=matrix)
#     print(q.q)
#     Quaternion.normalize(q.q)
#     print(q.q)
#     return q
#
# def constructPFromParameters(parameterArr):
#     c = numpy.asarray(parameterArr[0:3])
#     quaternion = numpy.array(parameterArr[3:7])
#     print(quaternion)
#     # quaternionLength = (quaternion[0] ** 2 + quaternion[1] ** 2 + quaternion[2] ** 2 + quaternion[3] ** 2) ** 0.5
#     # quaternion /= quaternionLength
#     r = util.quaternion_to_matrix(quaternion)
#     r = r[0:3, 0:3]
#     k = numpy.array([[parameterArr[7], 0, parameterArr[8]], [0, parameterArr[7], parameterArr[9]], [0, 0, parameterArr[10]]])
#     # reconstruct p
#     p = numpy.dot(k, r)
#     c = -numpy.dot(p, c)
#     p = numpy.append(p, [[c[0]], [c[1]], [c[2]]], 1)
#     print(p, c, k, r)
#     return p
#
#
# def constrainedError(parameterArr, correspondences):
#     p = constructPFromParameters(parameterArr)
#     return reprojectionError(p, correspondences)
#
#
# def nonLinearOptimizationConstrained(p, correspondences):
#     # extract optimization parameters
#     c, k, r = extractCameraParameters(p)
#     print(p, c, k, r)
#     quat = util.quaternion_from_matrix(r)
#     print(quat)
#     # optimization parameters:translation(x,y,z),rotation in quaternions(w,i,j,k), focal length (f), focal centre (fx,fy), special k
#     pflat = numpy.array([c[0], c[1], c[2], quat.q[0], quat.q[1], quat.q[2], quat.q[3], (k[0, 0] + k[1, 1]) / 2, k[0, 2], k[1, 2], k[2, 2]])
#     # p = optimize.leastsq(constrainedError, pflat, correspondences)
#     p = constructPFromParameters(pflat)
#
#     c, k, r = extractCameraParameters(p)
#     print("optimized p")
#     print(p)
#     return p
