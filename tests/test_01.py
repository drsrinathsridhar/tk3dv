########## Test common
import tk3dv
from tk3dv import common as tkc
from tk3dv.common import utilities as tkcu

########## Different ways of calling libs
print('Epoch time:', tk3dv.common.utilities.getCurrentEpochTime())
print('Epoch time:', tkc.utilities.getCurrentEpochTime())
print('Epoch time:', tkcu.getCurrentEpochTime())

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

########## Test ptTools
from tk3dv import ptTools

print('ptTools version:', tk3dv.ptTools.__version__)
print('ptTools version:', ptTools.__version__)
