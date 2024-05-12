import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.wrappers.scikit_learn import KerasClassifier

import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

def create_model():
    model = Sequential()

    model.add(Conv2D(filters= 64, kernel_size= (5, 5), activation= 'relu', input_shape = (224,224,3)))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters= 32, kernel_size= (3,3), activation= 'relu', kernel_regularizer= 'l2'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters= 16, kernel_size=(7,7), activation= 'relu', kernel_regularizer= 'l2'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters= 8, kernel_size=(5,5), activation= 'relu', kernel_regularizer= 'l2'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer= 'l2'))
    model.add(Dense(64, activation= 'relu', kernel_regularizer= 'l2'))
    model.add(Dense(len(config.CLASS_NAMES), activation= 'softmax'))
    opt= tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss = config.LOSS_FUCTION, optimizer= opt,metrics=config.MODEL_METRICS)

    return model

clf = KerasClassifier(build_fn=create_model, verbose=0)