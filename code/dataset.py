import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from torchnlp.word_to_vector import FastText, GloVe 
import json
import pickle
import string
from tqdm import tqdm
from utils import *


# instance attributes = (review_input,review_labels,book_id,user_id,timestamp,rating)
class SpoilerDataset(Dataset):
    def __init__(self, filename,
                 word_to_id_filename="word_to_id.pickle",id_to_word_filename="id_to_word.pickle",
                 book_to_id_filename="book_to_id.pickle", user_to_id_filename="user_to_id.pickle", load=True):
        super().__init__()
        self.filename = filename
        self.word_to_id = load_pickle(word_to_id_filename)
        self.id_to_word = load_pickle(id_to_word_filename)
        self.book_to_id = load_pickle(book_to_id_filename)
        self.user_to_id = load_pickle(user_to_id_filename)
        self.data = None
        if load:
            result, obj = safe_load_pickle(self.generate_processed_name())
            if result:
                self.data = obj
                return
        print("Generating ...")
        self.create_dataset()
                
        
    
    @classmethod
    def load_dataset(load_dataset):
        self.data = load_pickle(load_dataset)
    
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
                sentence_word_ids = [self.word_to_id.get(clean_word(word),self.word_to_id["<unk>"]) for word in sentence.split()]
                review_input.append(torch.tensor(sentence_word_ids,dtype=torch.long,requires_grad=False))
                review_labels.append(label) 
            instance = (review_input,review_labels,book_id,user_id,timestamp,rating)
            self.data[counter] = instance
            counter += 1
        save_pickle(self.data, self.generate_processed_name())
    
    def load_pretrained_embeddings(self, dim):
        vectors = GloVe(name="6B",dim=dim)
        return torch.stack([vectors[self.id_to_word[ind]] for ind in tqdm(range(len(self.id_to_word)))])
        
