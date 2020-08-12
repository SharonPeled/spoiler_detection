import string
import pickle
import os


SPECIAL_CHARACTER = "<unk>"
DATASET_PATH = '../data/goodreads_reviews_spoiler.json'
SAMPLE_DATASET_PATH = '../data/dataset_sample.json'
DATA_DIR = '../data/'
TEST_RATIO = 0.2
VALID_RATIO = 0.1
SUFFIX_ALL = "" # "_small"
PROCESSED_SUFFIX = "_processed"
VOCAB_SIZE = 20000
IDF_SMOOTHING = 1
# TOTAL_SIZE = 
# VERSION_SUFFIX = 


def add_suffix(filename):
    if len(filename.split(".")) == 1:
        return filename + SUFFIX_ALL
    if len(filename.split(".")) == 2:
        return filename.split(".")[0] + SUFFIX_ALL + "." + filename.split(".")[1]


def clean_word(word):
    # TODO: may need to change this for the pretrained embedding
    return word.translate(str.maketrans('', '', string.punctuation)).lower() # removing punctuation


def save_pickle(obj, filename):
    filename = add_suffix(filename)
    with open(os.path.join(DATA_DIR, filename), 'wb') as file:
        pickle.dump(obj, file)

        
def load_pickle(filename):
    filename = add_suffix(filename)
    with open(os.path.join(DATA_DIR, filename), 'rb') as file:
        return pickle.load(file)
    
    
def safe_load_pickle(filename):
    filename = add_suffix(filename)
    if not os.path.isfile(os.path.join(DATA_DIR, filename)):
        return False, None
    print(f"Loading existing {filename} .")
    return True, load_pickle(filename)


def save_model(model, version):
    save_pickle(model,f"model_{version}")

    
def map_to_not_in_vocabulary(size):
    return size 
