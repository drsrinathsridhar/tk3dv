import OpenGL.GL as gl
import numpy as np
import ctypes

def drawAxes(Length=100.0, LineWidth=5.0, Color=None):
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()

    gl.glPushAttrib(gl.GL_LINE_BIT)
    gl.glLineWidth(LineWidth)
    gl.glBegin(gl.GL_LINES)
    if Color is None:
        gl.glColor3f(1.0, 0.0, 0.0)
    else:
        gl.glColor3fv(Color)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(Length, 0.0, 0.0)

    if Color is None:
        gl.glColor3f(0.0, 1.0, 0.0)
    else:
        gl.glColor3fv(Color)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, Length, 0.0)

    if Color is None:
        gl.glColor3f(0.0, 0.0, 1.0)
    else:
        gl.glColor3fv(Color)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, Length)
    gl.glEnd()

    gl.glPopAttrib()
    gl.glPopMatrix()

def makeOpenGLMatrices(Intrinsics, ImageShape):
    # Returns appropriate ModelView and Projection matrices
    # Only intrinsics supported for now
    if(Intrinsics.shape[0] != 3 and Intrinsics.shape[1] != 3):
        raise Exception('Intrinsics matrix is not 3x3.')

    Width = ImageShape[1]
    Height = ImageShape[0]

    K = np.identity(4)
    K[0, 0:-1] = Intrinsics[0, :]
    K[1, 0:-1] = Intrinsics[1, :]
    K[2, 0:-1] = Intrinsics[2, :]
    # print(K)

    M = np.zeros(4)
    # TODO: Handle extrinsics
    OGLModelviewMatrix = M
    OGLModelviewMatrix[2, :] *= -1.0

    f = K[1, 1] * 2.0 / Height
    a = f / K[0, 0] * Width / 2.0
    Cx = K[0, 2]
    Cy = K[1, 2]

    cnear = 1.0
    cfar = 10000.0
    cleft = a * cnear / f * 2.0 * (-Cx / Width)
    cright = a * cnear / f * 2.0 * (1.0 - Cx / Width)
    ctop = cnear / f * 2.0 * (1.0 - Cy / Height)
    cbottom = cnear / f * 2.0 * (-Cy / Height)

    OGLProjectionMatrix = np.zeros((4, 4))
    OGLProjectionMatrix[0, 0] = (2 * cnear) / (cright - cleft)
    OGLProjectionMatrix[1, 1] = -(2 * cnear) / (ctop - cbottom)
    OGLProjectionMatrix[2, 2] = -(cfar + cnear) / (cfar - cnear)

    OGLProjectionMatrix[0, 2] = (cright + cleft) / (cright - cleft)
    OGLProjectionMatrix[1, 2] = -(ctop + cbottom) / (ctop - cbottom)
    OGLProjectionMatrix[3, 2] = -1

    OGLProjectionMatrix[2, 3] = -(2 * cfar * cnear) / (cfar - cnear)

    # print('[ P ]:\n', OGLProjectionMatrix)
    # print('[ M ]:\n', OGLModelviewMatrix)

    return OGLModelviewMatrix, OGLProjectionMatrix

def activateCamera(Intrinsics, ImageShape):
    MV, P = makeOpenGLMatrices(Intrinsics, ImageShape)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadMatrixf(P.transpose()) # Row major, so transpose

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadMatrixf(MV.transpose()) # Row major, so transpose

    # Somehow this function (in conjuction with MakeOpenGLMatrices) does not look right.
    # Adding this rotation makes it right
    gl.glRotatef(180.0, 0.0, 0.0, 1.0)


# TODO: Need to fix order of lines
UNITCUBE_V=[
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0, 1, 1,
        1, 1, 1,
        1, 0, 1,
        0, 0, 1,
]
UNITCUBE_C=[
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0, 1, 1,
        1, 1, 1,
        1, 0, 1,
        0, 0, 1,
        ]
# UNITCUBE_I=[
#         0, 1, 2, 2, 3, 0,
#         0, 3, 4, 4, 7, 0,
#         4, 7, 6, 6, 5, 4,
#         0, 7, 6, 6, 1, 0,
#         1, 6, 5, 5, 2, 1,
#         3, 4, 5, 5, 2, 3,
#         ]
UNITCUBE_I=[
        0, 1, 2, 2, 3, 0,
        0, 3, 4, 4, 7, 0,
        4, 7, 6, 6, 5, 4,
        0, 7, 6, 6, 1, 0,
        1, 6, 5, 5, 2, 1,
        3, 4, 5, 5, 2, 3,
        ]
# UNITCUBE_V=[
#         0, 0, 0,
#         1, 0, 0,
#         1, 1, 0,
#         0, 1, 0,
#         0, 0, 1,
#         1, 0, 1,
#         1, 1, 1,
#         0, 1, 1,
# ]
# UNITCUBE_C=[
#         0, 0, 0,
#         1, 0, 0,
#         1, 1, 0,
#         0, 1, 0,
#         0, 0, 1,
#         1, 0, 1,
#         1, 1, 1,
#         0, 1, 1,
#         ]
# UNITCUBE_I=[
#         0, 1, 2, 2, 3, 0,
#         0, 4, 5, 5, 1, 0,
#         1, 5, 6, 6, 2, 1,
#         2, 6, 7, 7, 3, 2,
#         3, 7, 4, 4, 0, 3,
#         4, 7, 6, 6, 5, 4,
#         ]

UNITCUBE_BUF=None
def createUnitCubeVBO():
    global UNITCUBE_BUF
    UNITCUBE_BUF = gl.glGenBuffers(3)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, UNITCUBE_BUF[0])
    gl.glBufferData(gl.GL_ARRAY_BUFFER,
            len(UNITCUBE_V)*4,  # byte size
            (ctypes.c_float*len(UNITCUBE_V))(*UNITCUBE_V),
            gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, UNITCUBE_BUF[1])
    gl.glBufferData(gl.GL_ARRAY_BUFFER,
            len(UNITCUBE_C)*4, # byte size
            (ctypes.c_float*len(UNITCUBE_C))(*UNITCUBE_C),
            gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, UNITCUBE_BUF[2])
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
            len(UNITCUBE_I)*4, # byte size
            (ctypes.c_uint*len(UNITCUBE_I))(*UNITCUBE_I),
            gl.GL_STATIC_DRAW)


def drawUnitWireCube(lineWidth=1.0, isRainbow=False, WireColor=(1, 1, 1)):
    # Draws a cube of size 1 centered at 0.5, 0.5, 0.5
    gl.glPushAttrib(gl.GL_LINE_WIDTH)
    gl.glLineWidth(lineWidth)

    gl.glColor3f(WireColor[0], WireColor[1], WireColor[2])

    # METHOD 1: Slow
    for i in range(0, len(UNITCUBE_I), 3): # Each face
        gl.glBegin(gl.GL_LINE_STRIP)
        index=UNITCUBE_I[i]*3
        if isRainbow:
            gl.glColor3f(*UNITCUBE_C[index:index+3])
        gl.glVertex3f(*UNITCUBE_V[index:index+3])

        index=UNITCUBE_I[i+1]*3
        if isRainbow:
            gl.glColor3f(*UNITCUBE_C[index:index+3])
        gl.glVertex3f(*UNITCUBE_V[index:index+3])

        index=UNITCUBE_I[i+2]*3
        if isRainbow:
            gl.glColor3f(*UNITCUBE_C[index:index+3])
        gl.glVertex3f(*UNITCUBE_V[index:index+3])
        gl.glEnd()

    # METHOD 2: VBOs
    # if UNITCUBE_BUF is None:
    #     createUnitCubeVBO()
    #
    # gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    # gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, UNITCUBE_BUF[0])
    # gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
    # if isRainbow:
    #     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, UNITCUBE_BUF[1])
    #     gl.glColorPointer(3, gl.GL_FLOAT, 0, None)
    # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, UNITCUBE_BUF[2])
    # gl.glDrawElements(gl.GL_LINE_STRIP, len(UNITCUBE_I), gl.GL_UNSIGNED_INT, None)
    # gl.glDisableClientState(gl.GL_COLOR_ARRAY)
    # gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    gl.glPopAttrib()

def drawUnitCube(isRainbow=False, Color=(1, 1, 1), Alpha=1.0):
    gl.glPushAttrib(gl.GL_COLOR_BUFFER_BIT)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)

    # METHOD 1: Slow for multiple cubes
    # Drawing CCW
    gl.glBegin(gl.GL_QUADS)
    if isRainbow == False:
        gl.glColor4f(Color[0], Color[1], Color[2], Alpha)

    for i in range(0, 2):
        # Bottom and Top
        if isRainbow:
            gl.glColor4f(0.0, 0.0, i, Alpha)
        gl.glVertex3f(0.0, 0.0, i)
        if isRainbow:
            gl.glColor4f(1.0, 0.0, i, Alpha)
        gl.glVertex3f(1.0, 0.0, i)
        if isRainbow:
            gl.glColor4f(1.0, 1.0, i, Alpha)
        gl.glVertex3f(1.0, 1.0, i)
        if isRainbow:
            gl.glColor4f(0.0, 1.0, i, Alpha)
        gl.glVertex3f(0.0, 1.0, i)

        # Right and Left
        if isRainbow:
            gl.glColor4f(i, 0.0, 0.0, Alpha)
        gl.glVertex3f(i, 0.0, 0.0)
        if isRainbow:
            gl.glColor4f(i, 1.0, 0.0, Alpha)
        gl.glVertex3f(i, 1.0, 0.0)
        if isRainbow:
            gl.glColor4f(i, 1.0, 1.0, Alpha)
        gl.glVertex3f(i, 1.0, 1.0)
        if isRainbow:
            gl.glColor4f(i, 0.0, 1.0, Alpha)
        gl.glVertex3f(i, 0.0, 1.0)

        # Front and Back
        if isRainbow:
            gl.glColor4f(0.0, i, 0.0, Alpha)
        gl.glVertex3f(0.0, i, 0.0)
        if isRainbow:
            gl.glColor4f(1.0, i, 0.0, Alpha)
        gl.glVertex3f(1.0, i, 0.0)
        if isRainbow:
            gl.glColor4f(1.0, i, 1.0, Alpha)
        gl.glVertex3f(1.0, i, 1.0)
        if isRainbow:
            gl.glColor4f(0.0, i, 1.0, Alpha)
        gl.glVertex3f(0.0, i, 1.0)
    gl.glEnd()

    # # METHOD 2: VBOs
    # if UNITCUBE_BUF is None:
    #     createUnitCubeVBO()
    #
    # gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    # gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, UNITCUBE_BUF[0])
    # gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
    # if isRainbow:
    #     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, UNITCUBE_BUF[1])
    #     gl.glColorPointer(3, gl.GL_FLOAT, 0, None)
    # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, UNITCUBE_BUF[2])
    # gl.glDrawElements(gl.GL_TRIANGLES, len(UNITCUBE_I), gl.GL_UNSIGNED_INT, None)
    # gl.glDisableClientState(gl.GL_COLOR_ARRAY)
    # gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    gl.glPopAttrib()


def drawCheckerBoard(floorSize, squareWidthInPixel, squareHeightInPixel, SceneHeight):
    mat_specular = np.array([1.0, 1.0, 1.0, 1.0])
    mat_shininess = np.array([128])
    light_position = np.array([0.0, 3.0, 0.0, 0.0])
    colorBlack = np.array([0.8, 0.8, 0.8, 1.0])
    colorWhite = np.array([0.1, 0.1, 0.1, 1.0])

    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, mat_specular)

    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)

    color = False
    for x in range(-floorSize, floorSize + 1, squareHeightInPixel):
        for y in range(-floorSize, floorSize + 1, squareHeightInPixel):
            x1 = x
            y1 = y + squareHeightInPixel
            x2 = x + squareWidthInPixel
            y2 = y + squareHeightInPixel
            x3 = x + squareWidthInPixel
            y3 = y
            x4 = x
            y4 = y

            if (color == True):
                gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, colorWhite)
                gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, colorWhite)
            else:
                gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, colorBlack)
                gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, colorBlack)
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, mat_shininess)
            color = not color

            gl.glBegin(gl.GL_QUADS)

            gl.glNormal3f(0.0, 1.0, 0.0)
            gl.glVertex3f(x1, -SceneHeight, y1)

            gl.glNormal3f(0.0, 1.0, 0.0)
            gl.glVertex3f(x2, -SceneHeight, y2)

            gl.glNormal3f(0.0, 1.0, 0.0)
            gl.glVertex3f(x3, -SceneHeight, y3)

            gl.glNormal3f(0.0, 1.0, 0.0)
            gl.glVertex3f(x4, -SceneHeight, y4)

            gl.glEnd()

    gl.glDisable(gl.GL_LIGHT0)
    gl.glDisable(gl.GL_LIGHTING)

g_isSetupTextures = False
g_TextureID = 0
def setupTextures():
    global g_TextureID
    gl.glGenTextures(1, g_TextureID)
    gl.glBindTexture(gl.GL_TEXTURE_2D, g_TextureID)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)

    gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    global g_isSetupTextures
    g_isSetupTextures = True

def drawImage(Image):
    if Image is None:
        return

    if Image.dtype is not np.dtype('uint8'):
        return

    global g_isSetupTextures
    if g_isSetupTextures == False:
        setupTextures()

    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glPushMatrix();

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    gl.glOrtho(0, 1, 1, 0, 1, -1)

    # NOTE: Assuming texture is already bound
    gl.glBindTexture(gl.GL_TEXTURE_2D, g_TextureID)
    GLImage = np.fromstring(Image.tostring(), np.uint8)
    if (len(Image.shape) == 2):
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_LUMINANCE, Image.shape[1], Image.shape[0], 0, gl.GL_LUMINANCE,
                        gl.GL_UNSIGNED_BYTE, GLImage)
    elif (len(Image.shape) == 3):
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, Image.shape[1], Image.shape[0], 0, gl.GL_BGR,
                        gl.GL_UNSIGNED_BYTE, GLImage)

    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2f(0, 1)
    gl.glVertex2f(0, 1)

    gl.glTexCoord2f(1, 1)
    gl.glVertex2f(1, 1)

    gl.glTexCoord2f(1, 0)
    gl.glVertex2f(1, 0)

    gl.glTexCoord2f(0, 0)
    gl.glVertex2f(0, 0)
    gl.glEnd()

    gl.glDisable(gl.GL_TEXTURE_2D)

    gl.glPopMatrix()
    gl.glEnable(gl.GL_DEPTH_TEST)
