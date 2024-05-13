import os
import sys
import pathlib

PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
with open(os.path.join(config.PACKAGE_ROOT,'VERSION')) as f:
    __version__ = f.read().strip()