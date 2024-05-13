import pytest 
from sklearn.model_selection import train_test_split
import numpy as np

from prediction_model.config import config
from prediction_model.processing.data_handling import load_train_dataset
from prediction_model.predict import generate_prediction_test

@pytest.fixture
def single_prediction():
    train_data, Y_data, labels = load_train_dataset(config.LABEL_FILE)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, Y_data, test_size= 0.1, shuffle=False)
    single_row = X_test[:1]
    Y_pred = generate_prediction_test(single_row)
    pred = labels['breed'][np.argmax(Y_pred[0])]
    result = {"Predicted" : pred}
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('Predicted'),str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('Predicted') in config.CLASS_NAMES