########## Test common
import tk3dv
from tk3dv import common as tkc
from tk3dv.common import utilities as tkcu

print(tk3dv.common.utilities.getCurrentEpochTime())
print(tkc.utilities.getCurrentEpochTime())
print(tkcu.getCurrentEpochTime())

print('common version:', tk3dv.common.__version__)
print('common version:', tkc.__version__)


########## Test nocstools
from tk3dv import nocstools as nt

print('nocstools version:', tk3dv.nocstools.__version__)
print('nocstools version:', nt.__version__)

########## Test nocstools
from tk3dv import pyEasel

print('pyEasel version:', tk3dv.pyEasel.__version__)
print('pyEasel version:', pyEasel.__version__)
