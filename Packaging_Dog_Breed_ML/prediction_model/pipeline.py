from sklearn.pipeline import Pipeline

import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.model import clf

classification_pipeline = Pipeline(
    [
        ('ModelTrain', clf)
    ]
)
