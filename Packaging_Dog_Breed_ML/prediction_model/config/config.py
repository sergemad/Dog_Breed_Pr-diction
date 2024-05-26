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

ALL_CLASS = ['boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever',
       'bedlington_terrier', 'borzoi', 'basenji', 'scottish_deerhound',
       'shetland_sheepdog', 'walker_hound', 'maltese_dog',
       'norfolk_terrier', 'african_hunting_dog',
       'wire-haired_fox_terrier', 'redbone', 'lakeland_terrier', 'boxer',
       'doberman', 'otterhound', 'standard_schnauzer',
       'irish_water_spaniel', 'black-and-tan_coonhound', 'cairn',
       'affenpinscher', 'labrador_retriever', 'ibizan_hound',
       'english_setter', 'weimaraner', 'giant_schnauzer', 'groenendael',
       'dhole', 'toy_poodle', 'border_terrier', 'tibetan_terrier',
       'norwegian_elkhound', 'shih-tzu', 'irish_terrier', 'kuvasz',
       'german_shepherd', 'greater_swiss_mountain_dog', 'basset',
       'australian_terrier', 'schipperke', 'rhodesian_ridgeback',
       'irish_setter', 'appenzeller', 'bloodhound', 'samoyed',
       'miniature_schnauzer', 'brittany_spaniel', 'kelpie', 'papillon',
       'border_collie', 'entlebucher', 'collie', 'malamute',
       'welsh_springer_spaniel', 'chihuahua', 'saluki', 'pug', 'malinois',
       'komondor', 'airedale', 'leonberg', 'mexican_hairless',
       'bull_mastiff', 'bernese_mountain_dog',
       'american_staffordshire_terrier', 'lhasa', 'cardigan',
       'italian_greyhound', 'clumber', 'scotch_terrier', 'afghan_hound',
       'old_english_sheepdog', 'saint_bernard', 'miniature_pinscher',
       'eskimo_dog', 'irish_wolfhound', 'brabancon_griffon',
       'toy_terrier', 'chow', 'flat-coated_retriever', 'norwich_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'english_foxhound', 'gordon_setter', 'siberian_husky',
       'newfoundland', 'briard', 'chesapeake_bay_retriever',
       'dandie_dinmont', 'great_pyrenees', 'beagle', 'vizsla',
       'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet',
       'sealyham_terrier', 'standard_poodle', 'keeshond',
       'japanese_spaniel', 'miniature_poodle', 'pomeranian',
       'curly-coated_retriever', 'yorkshire_terrier', 'pembroke',
       'great_dane', 'blenheim_spaniel', 'silky_terrier',
       'sussex_spaniel', 'german_short-haired_pointer', 'french_bulldog',
       'bouvier_des_flandres', 'tibetan_mastiff', 'english_springer',
       'cocker_spaniel', 'rottweiler']

#FEATURES = []

#NUM_FEATURES = []

LEARNING_RATE = 0.001

LOSS_FUCTION = 'categorical_crossentropy'

MODEL_METRICS = ['accuracy']

EPOCHS = 10

BATCH_SIZE = 128