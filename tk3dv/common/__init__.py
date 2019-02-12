# Written by Srinath Sridhar (http://srinathsridhar.com)

# Released under the MIT License, see LICENSE.txt
import os, sys
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '.'))

import utilities, drawing, defines

__version__= defines.__version__
