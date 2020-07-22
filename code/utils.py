import string
import pickle
import os


SPECIAL_CHARACTER = "<unk>"
DATASET_PATH = '../data/goodreads_reviews_spoiler.json'
DATA_DIR = '../data/'
TEST_RATIO = 0.2
VALID_RATIO = 0.2


def clean_word(word):
    # TODO: may need to change this for the pretrained embedding
    return word.translate(str.maketrans('', '', string.punctuation)) # removing punctuation


def save_pickle(obj, filename):
    with open(os.path.join(DATA_DIR, filename), 'wb') as file:
        pickle.dump(obj, file)

        
def load_pickle(filename):
    with open(os.path.join(DATA_DIR, filename), 'rb') as file:
        return pickle.load(file)