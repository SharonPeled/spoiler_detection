import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from torchnlp.word_to_vector import FastText, GloVe 
import json
import pickle
import string
from tqdm import tqdm
from utils import *
import numpy as np

# instance attributes = (review_input,review_labels,book_id,user_id,timestamp,rating)
class SpoilerDataset(Dataset):
    def __init__(self, filename,
                 word_to_id_filename="word_to_id.pickle",id_to_word_filename="id_to_word.pickle",
                 book_to_id_filename="book_to_id.pickle", user_to_id_filename="user_to_id.pickle",
                 book_reviews_count_filename="book_reviews_count.pickle",
                 word_per_book_count_filename="word_per_book_count.pickle", load=True):
        super().__init__()
        self.filename = filename
        self.word_to_id = load_pickle(word_to_id_filename)
        self.id_to_word = load_pickle(id_to_word_filename)
        self.book_to_id = load_pickle(book_to_id_filename)
        self.user_to_id = load_pickle(user_to_id_filename)
        self.book_reviews_count = load_pickle(book_reviews_count_filename)
        self.word_per_book_count = load_pickle(word_per_book_count_filename)
        self.data = None
        if load:
            result, obj = safe_load_pickle(self.generate_processed_name())
            if result:
                self.data = obj
                return
        print("Generating ...")
        self.create_dataset()
    
    def generate_processed_name(self):
        return self.filename.split(".")[0] + PROCESSED_SUFFIX + "." + self.filename.split(".")[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
    
    def get_num_books(self):
        return len(self.book_to_id)
    
    def get_num_users(self):
        return len(self.user_to_id)
    
    def create_dataset(self):
        samples = load_pickle(self.filename)
        self.data = {}
        counter = 0
        for data_dict in tqdm(samples):
            timestamp = data_dict['timestamp']
            rating = data_dict['rating']
            book_id = self.book_to_id[data_dict['book_id']]
            user_id = self.user_to_id[data_dict['user_id']]
            review_sentences = data_dict['review_sentences']
            review_input = []
            review_labels = []
            for label, sentence in review_sentences:
                if not sentence:
                    continue
                sentence_word_ids = [self.word_to_id[clean_word(word)] for word in sentence.split()]
#                 sentence_word_ids = [self.word_to_id.get(clean_word(word),self.word_to_id[SPECIAL_CHARACTER]) for word in sentence.split()]
                review_input.append(torch.tensor(sentence_word_ids,dtype=torch.long,requires_grad=False))
                review_labels.append(label) 
            instance = (review_input,review_labels,book_id,user_id,timestamp,rating)
            self.data[counter] = instance
            counter += 1
        save_pickle(self.data, self.generate_processed_name())
    
    def load_pretrained_embeddings(self, dim):
        vectors = GloVe(name="6B",dim=dim)
        embeddings = torch.stack([vectors[self.id_to_word[ind]] for ind in tqdm(range(len(self.id_to_word)))])
        embeddings = embeddings.type(torch.float64)
        return embeddings
    
    def get_tf_idf_features_tensor(self, sentence_batch, book_id):
        return torch.tensor(
            [
                [self.generate_word_features(word_id, book_id) for word_id in sentence]
                for sentence in sentence_batch
            ]
            , dtype=torch.float64
        )
    
    def generate_word_features(self, word_id, book_id):
        word_id = word_id.item()
        book_id = book_id.item()
        word = self.id_to_word[word_id]
        if word == SPECIAL_CHARACTER:
            return [0.0, 0.0, 0.0]
        num_reviews = self.book_reviews_count[book_id]
        num_tot_books = len(self.book_reviews_count)
        word_occ_in_reviews = self.word_per_book_count[word][book_id]
        tot_book_word_occ = len(self.word_per_book_count[word])
        DF = word_occ_in_reviews/num_reviews
        IIF = np.log((IDF_SMOOTHING+tot_book_word_occ)/(IDF_SMOOTHING+num_tot_books))
        return [DF, IIF, DF*IIF]
