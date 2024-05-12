import os
import pandas as pd
import numpy as np
import joblib

import tensorflow as tf
from PIL import Image

from sklearn.calibration import label_binarize

from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config

#Load the dataset
def load_train_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    labels_all = pd.read_csv(filepath) 
    labels = labels_all[(labels_all[config.TARGET]).isin(config.CLASS_NAMES)]
    labels = labels.reset_index()

    #creating numpy matrix with zeros
    X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32') # 3 -> RGB

    Y_data = label_binarize(labels[config.TARGET], classes= config.CLASS_NAMES)

    train_path = os.path.join(config.DATAPATH,config.TRAIN_FILE)

    # Reading and converting image to numpy array and normalizeing dataset
    for i in range(len(labels)): #tqdm progress bar of your loop
        img_filename = f"{labels['id'][i]}.jpg"
        img_file_path = f"{train_path}/{img_filename}"
        img = Image.open(img_file_path)
        #img = image.load_img(train_path.join('/%s.jpg' % labels['id'][i]), target_size=(224, 224))
        #img = image.load_img('archive/train/%s.jpg' % labels['id'][i],target_size=(224, 224)) # '%s' % name or '%i' % value ...
        #img = image.img_to_array(img)
        img = img.resize((224, 224)) # Resize the image to target size
        img_array = np.array(img)
        x = np.expand_dims(img_array.copy(),axis=0) # expand the dimension or size to (1, 224, 224, 3) instead of (224, 224, 3)
        X_data[i] = x / 255.0 # divide by 255 because we want value of RGB to be between 0 and 1

    return X_data, Y_data, labels

# Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"The Model has been saved under the name {config.MODEL_NAME}")

# Deserialization
def load_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print(f"The Model has been loaded")
    return model_loaded