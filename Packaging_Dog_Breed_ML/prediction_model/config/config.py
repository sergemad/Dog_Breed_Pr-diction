import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE = 'train'
TEST_FILE = 'test'
LABEL_FILE = 'labels.csv'

MODEL_NAME = 'Dog_breed_classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'breed'

CLASS_NAMES = ['scottish_deerhound','maltese_dog','bernese_mountain_dog']

#FEATURES = []

#NUM_FEATURES = []

LEARNING_RATE = 0.001

LOSS_FUCTION = 'categorical_crossentropy'

MODEL_METRICS = ['accuracy']

