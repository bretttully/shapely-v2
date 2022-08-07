from distutils.version import LooseVersion

import shapely

SHAPELY_GE_20 = str(shapely.__version__) >= LooseVersion("2.0")
