import warnings
warnings.warn('Please import Universe class from universe module : from brownpy import Universe', 
              DeprecationWarning)

import sys

# make sure bar is in sys.modules
from brownpy import universe
# link this module to universe
sys.modules[__name__] = sys.modules['brownpy.universe']

