## Fix for Mac OS Big Sur - Starts
try:
    import OpenGL as OpenGL
    try:
        import OpenGL.GL   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        print('Drat, patching for Big Sur')  # https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = new_util_find_library
except ImportError:
    pass
## Fix for Mac OS Big Sur - Ends
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.arrays.vbo as glvbo
import numpy as np
import math, sys

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


QUADRIC = glu.gluNewQuadric()
# glu.gluDeleteQuadric(QUADRIC)

def drawSolidSphere(radius=1.0, slices=16, stacks=16, Color=None):
    if (Color != None):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA) # Orig
        gl.glEnable(gl.GL_BLEND)

        gl.glColor4fv(Color)
    else:
        gl.glColor3f(0.0, 0.0, 0.0)

    glu.gluQuadricDrawStyle(QUADRIC, glu.GLU_FILL)
    glu.gluSphere(QUADRIC, radius, slices, stacks)
    
def drawWireSphere(radius=1.0, slices=16, stacks=16, Color=None):
    if (Color != None):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA) # Orig
        gl.glEnable(gl.GL_BLEND)

        gl.glColor4fv(Color)
    else:
        gl.glColor3f(0.0, 0.0, 0.0)

    glu.gluQuadricDrawStyle(QUADRIC, glu.GLU_LINE)
    glu.gluSphere(QUADRIC, radius, slices, stacks)

def drawCylinder(Start=np.array([0, 0, 0]), End=np.array([1.0, 0.0, 0.0]), Radius1=1.0, Radius2=1.0, Color=None):
    if type(Start) is not np.ndarray or type(End) is not np.ndarray:
        raise RuntimeError('Start and End need to be Numpy arrays.')

    Direction = End - Start
    Length = np.linalg.norm(Direction)
    if (Length <= 0.0):
        return
    Direction = Direction / Length

    # Find out the axis of rotation and angle of rotation to rotate the
    # gluCylinder (oriented along the z axis) into the desired direction
    Z = np.array([0., 0., 1.])
    Axis = np.cross(Z, Direction)
    Angle = math.acos(np.dot(Z, Direction)) * (180. / math.pi) # Should be degrees

    gl.glPushMatrix()
    gl.glTranslate(Start[0], Start[1], Start[2])
    gl.glRotate(Angle, Axis[0], Axis[1], Axis[2])

    # Next the 6 faces
    if (Color != None):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA) # Orig
        gl.glEnable(gl.GL_BLEND)

        gl.glColor4fv(Color)
    else:
        gl.glColor3f(0.0, 0.0, 0.0)

    # Draw cylinder
    # Bottom:
    glu.gluQuadricOrientation(QUADRIC, glu.GLU_INSIDE)
    glu.gluDisk(QUADRIC, 0, Radius1, 16, 1)

    glu.gluQuadricOrientation(QUADRIC, glu.GLU_OUTSIDE)
    glu.gluCylinder(QUADRIC, Radius1, Radius2, Length, 16, 1)

    # Top:
    gl.glTranslatef(0, 0, Length)
    glu.gluQuadricOrientation(QUADRIC, glu.GLU_OUTSIDE)
    glu.gluDisk(QUADRIC, 0, Radius2, 16, 1)

    gl.glPopMatrix()

def drawCone(Start=np.array([0, 0, 0]), End=np.array([1.0, 0.0, 0.0]), Radius1=1, Radius2=0.5, Color=None):
    drawCylinder(Start, End, Radius1, Radius2, Color)

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
UNITCUBE_I=[
        0, 1, 2, 2, 3, 0,
        0, 3, 4, 4, 7, 0,
        4, 7, 6, 6, 5, 4,
        0, 7, 6, 6, 1, 0,
        1, 6, 5, 5, 2, 1,
        3, 4, 5, 5, 2, 3,
        ]

def drawUnitWireCube(lineWidth=1.0, isRainbow=False, WireColor=(1, 1, 1)):
    # Draws a cube of size 1 centered at 0.5, 0.5, 0.5
    gl.glPushAttrib(gl.GL_LINE_WIDTH)
    gl.glLineWidth(lineWidth)

    gl.glColor3f(WireColor[0], WireColor[1], WireColor[2])

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

    gl.glPopAttrib()

UNITFRUSTUM_V=[
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0.25, 0.75, 1,
        0.75, 0.75, 1,
        0.75, 0.25, 1,
        0.25, 0.25, 1,
]
UNITFRUSTUM_C=[
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0.25, 0.75, 1,
        0.75, 0.75, 1,
        0.75, 0.25, 1,
        0.25, 0.25, 1,
        ]
UNITFRUSTUM_I=[
        0, 1, 2, 2, 3, 0,
        0, 3, 4, 4, 7, 0,
        4, 7, 6, 6, 5, 4,
        0, 7, 6, 6, 1, 0,
        1, 6, 5, 5, 2, 1,
        3, 4, 5, 5, 2, 3,
        ]

def drawUnitWireFrustum(lineWidth=1.0, isRainbow=False, WireColor=(1, 1, 1)):
    # Draws a frustum of size 1 and 0.5 centered at 0.5, 0.5, 0.5
    gl.glPushAttrib(gl.GL_LINE_WIDTH)
    gl.glLineWidth(lineWidth)

    gl.glColor3f(WireColor[0], WireColor[1], WireColor[2])

    for i in range(0, len(UNITFRUSTUM_I), 3): # Each face
        gl.glBegin(gl.GL_LINE_STRIP)
        index=UNITFRUSTUM_I[i]*3
        if isRainbow:
            gl.glColor3f(*UNITFRUSTUM_C[index:index+3])
        gl.glVertex3f(*UNITFRUSTUM_V[index:index+3])

        index=UNITFRUSTUM_I[i+1]*3
        if isRainbow:
            gl.glColor3f(*UNITFRUSTUM_C[index:index+3])
        gl.glVertex3f(*UNITFRUSTUM_V[index:index+3])

        index=UNITFRUSTUM_I[i+2]*3
        if isRainbow:
            gl.glColor3f(*UNITFRUSTUM_C[index:index+3])
        gl.glVertex3f(*UNITFRUSTUM_V[index:index+3])
        gl.glEnd()

    gl.glPopAttrib()


def getVBOs(V, VC, I):
    VBO_V = glvbo.VBO(V)
    VBO_VC = glvbo.VBO(VC)
    VBO_I = glvbo.VBO(I, target=gl.GL_ELEMENT_ARRAY_BUFFER)

    return VBO_V, VBO_VC, VBO_I


CB_V = np.zeros([0, 3], dtype=np.float32)  # Each point is a row
CB_VC = np.zeros([0, 4], dtype=np.float32)  # Each point is a row
CB_I = np.zeros([0, 1], dtype=np.int32)  # Each element is an index
CB_V_VBO = None
CB_VC_VBO = None
CB_I_VBO = None
CBFloorSize = 10000
CBSquareWidth = 1000
CBSquareHeight = 1000
CBSceneHeight = 1000
CBVBOBound = False
CB_isWire = False
CB_WireColor = np.array([0.1, 0.1, 0.1, 1.0])

def createCBData(floorSize, squareWidthInPixel, squareHeightInPixel, SceneHeight):
    global CBVBOBound, CB_V, CB_VC, CB_I, CB_V_VBO, CB_VC_VBO, CB_I_VBO, CB_isWire, CB_WireColor
    CBVBOBound = False
    colorBlack = np.array([0.8, 0.8, 0.8, 1.0])
    colorWhite = np.array([0.1, 0.1, 0.1, 1.0])
    isWhite = False
    Idx = 0
    CB_V = np.zeros([0, 3], dtype=np.float32)  # Each point is a row
    CB_VC = np.zeros([0, 4], dtype=np.float32)  # Each point is a row
    CB_I = np.zeros([0, 1], dtype=np.int32)  # Each element is an index

    for x in range(-floorSize, floorSize + 1, squareHeightInPixel):
        for y in range(-floorSize, floorSize + 1, squareWidthInPixel):
            CB_V = np.vstack([CB_V, np.array([x, -SceneHeight, y + squareHeightInPixel])])
            CB_V = np.vstack([CB_V, np.array([x + squareHeightInPixel, -SceneHeight, y + squareHeightInPixel])])
            CB_V = np.vstack([CB_V, np.array([x + squareHeightInPixel, -SceneHeight, y])])
            CB_V = np.vstack([CB_V, np.array([x, -SceneHeight, y])])

            CB_I = np.vstack([CB_I, np.array([Idx, Idx + 1, Idx + 2]).reshape(-1, 1)])
            CB_I = np.vstack([CB_I, np.array([Idx + 2, Idx + 3, Idx]).reshape(-1, 1)])
            Idx = Idx + 4

            for kk in range(0, 4):
                if CB_isWire:
                    CB_VC = np.vstack([CB_VC, CB_WireColor])
                else:
                    if (isWhite == True):
                        CB_VC = np.vstack([CB_VC, colorWhite])
                    else:
                        CB_VC = np.vstack([CB_VC, colorBlack])
            isWhite = not isWhite

    CB_V_VBO, CB_VC_VBO, CB_I_VBO = getVBOs(CB_V, CB_VC, CB_I)

def drawCheckerBoard(floorSize, squareWidthInPixel, squareHeightInPixel, SceneHeight, isWireFrame=False, LineWidth=3.0, wireColor=np.array([0, 0, 0, 1])):
    global CB_V, CB_VC, CB_I, CBFloorSize, CBSquareWidth, CBSquareHeight, CBVBOBound, CB_V_VBO, CB_VC_VBO, CB_I_VBO, CB_isWire, CB_WireColor
    if floorSize != CBFloorSize or squareHeightInPixel != CBSquareWidth or squareHeightInPixel != CBSquareHeight \
            or CBSceneHeight != SceneHeight or CB_V_VBO is None or CB_isWire != isWireFrame or np.linalg.norm(CB_WireColor - wireColor) > 0.01:
        CB_WireColor = wireColor
        CB_isWire = isWireFrame
        createCBData(floorSize, squareWidthInPixel, squareHeightInPixel, SceneHeight)

    if CBVBOBound == False:
        CB_V_VBO.bind()
        CB_VC_VBO.bind()
        CB_I_VBO.bind()
        CBVBOBound = True

    gl.glPushAttrib(gl.GL_POLYGON_BIT)
    gl.glPushAttrib(gl.GL_COLOR_BUFFER_BIT)
    gl.glPushAttrib(gl.GL_LINE_WIDTH)
    gl.glLineWidth(LineWidth)

    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)

    CB_V_VBO.bind()
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glVertexPointer(3, gl.GL_DOUBLE, 0, CB_V_VBO)
    CB_VC_VBO.bind()
    gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    gl.glColorPointer(4, gl.GL_DOUBLE, 0, CB_VC_VBO)

    CB_I_VBO.bind()
    if isWireFrame:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    gl.glDrawElements(gl.GL_TRIANGLES, int(len(CB_I_VBO)), gl.GL_UNSIGNED_INT, None)

    gl.glDisableClientState(gl.GL_COLOR_ARRAY)
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    gl.glPopAttrib()
    gl.glPopAttrib()
    gl.glPopAttrib()

def lightsOn():
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glEnable(gl.GL_NORMALIZE)
    gl.glEnable(gl.GL_LIGHTING)

    # TODO: No idea why light1 doesn't work when light0 is active. But just light1 works fine
    # light0_position = np.array([3, 3, 3, 0.0])
    # gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light0_position)
    light1_position = np.array([0, 3, 3, 0.0])
    gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, light1_position)
    # light2_position = np.array([3, 3, 3, 0.0])
    # gl.glLightfv(gl.GL_LIGHT2, gl.GL_POSITION, light2_position)

    # gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
    gl.glEnable(gl.GL_LIGHT0)
    gl.glEnable(gl.GL_LIGHT1)
    # gl.glEnable(gl.GL_LIGHT2)

def lightsOff():
    gl.glDisable(gl.GL_LIGHT0)
    gl.glDisable(gl.GL_LIGHT1)
    # gl.glDisable(gl.GL_LIGHT2)
    gl.glDisable(gl.GL_LIGHTING)
    
def enableDefaultMaterial()
    color = np.array([0.5, 0.6, 0.8, 1.0])
    mat_shininess = np.array([128])
    mat_specular = np.array([1.0, 1.0, 1.0, 1.0]) / 4
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, color)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, mat_specular)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SHININESS, mat_shininess)

def drawCheckerBoardOld(floorSize, squareWidthInPixel, squareHeightInPixel, SceneHeight):
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
