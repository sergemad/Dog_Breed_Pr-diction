import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline,load_train_dataset
import tensorflow as tf
from keras.preprocessing import image

classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_prediction_test(data_input):
    Y_pred = classification_pipeline.predict(data_input)
    return Y_pred


def generate_prediction():
    train_data, Y_data, labels = load_train_dataset(config.LABEL_FILE)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, Y_data, test_size= 0.1, shuffle=False)

    Y_pred = classification_pipeline.predict(X_test)

    result = {"Originally : ", labels['breed'][np.argmax(Y_test[1])],
              "Predicted : ", labels['breed'][np.argmax(Y_pred[1])]}
    
    print("Originally : ", labels['breed'][np.argmax(Y_test[1])])
    print("Predicted : ", labels['breed'][np.argmax(Y_pred[1])])
          
    
    return result

if __name__ == '__main__':
    generate_prediction()