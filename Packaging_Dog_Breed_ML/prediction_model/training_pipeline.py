import pandas as pd
import numpy as np

import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_train_dataset,save_pipeline
import prediction_model.pipeline as pipe
from sklearn.model_selection import train_test_split

def perform_training():

    train_data, Y_data, labels = load_train_dataset(config.LABEL_FILE)
    # Splitting the data set into training and testing data sets
    X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(train_data, Y_data, test_size= 0.1, shuffle=False)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size= 0.2) 
    pipe.classification_pipeline.fit(X_train, Y_train, batch_size= config.BATCH_SIZE, epochs= config.EPOCHS, validation_data= (X_val, Y_val))
    save_pipeline(pipe.classification_pipeline)

if __name__ == '__main__':
    perform_training()