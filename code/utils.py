import string
import pickle
import os


SPECIAL_CHARACTER = "<unk>"
DATASET_PATH = '../data/goodreads_reviews_spoiler.json'
DATA_DIR = '../data/'
TEST_RATIO = 0.2
VALID_RATIO = 0.2
PROCESSED_SUFFIX = "_processed"


def clean_word(word):
    # TODO: may need to change this for the pretrained embedding
    return word.translate(str.maketrans('', '', string.punctuation)).lower() # removing punctuation


def save_pickle(obj, filename):
    with open(os.path.join(DATA_DIR, filename), 'wb') as file:
        pickle.dump(obj, file)

        
def load_pickle(filename):
    with open(os.path.join(DATA_DIR, filename), 'rb') as file:
        return pickle.load(file)
    
    
def safe_load_pickle(filename):
    if not os.path.isfile(os.path.join(DATA_DIR, filename)):
        return False, None
    print(f"Loading existing {filename} .")
    return True, load_pickle(filename)


def save_model(model, version):
    save_pickle(model,f"model_{version}")
    