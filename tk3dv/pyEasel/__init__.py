# pyEasel written by Srinath Sridhar (http://srinathsridhar.com)
import sys, os
FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'py'))

import defines, Easel, EaselModule, GLViewer, pyEasel

__version__= defines.__version__
