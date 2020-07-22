import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
import json
import pickle
import string
from utils import *


# In[5]:


# instance attributes = (review_input_tensor,review_labels_tensor,bookd_id,user_id,timestamp,rating)


# In[46]:

# In[15]:


# instance attributes = (words_review_tensor,summary_review_tensor,label,review_date,movie_id,user_id)


        

class SpoilerDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH,
                 word_to_id_filename="word_to_id.pickle",id_to_word_filename=="id_to_word.pickle"):
        super().__init__()
        self.dataset_path = dataset_path
        self.word_to_id = load_pickle(word_to_id_filename)
        self.id_to_word = load_pickle(id_to_word_filename)
        self.create_dataset()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
    
    def create_dataset(self):
        data = {}
        counter = 0
        with open(self.dataset_path) as json_file:
            for line in json_file:
                data_dict = json.loads(line)
                user_id = data_dict['user_id']
                timestamp = data_dict['timestamp']
                rating = data_dict['rating']
                book_id = data_dict['book_id']
                
                review_input = []
                review_labels = []
                
                review_sentences = data_dict['review_sentences']

                for sentence_with_label in review_sentences:
                    sentence_input = []
                    sentence = sentence_with_label[1]
                    sentence_label = sentence_with_label[0]

                    cleaned_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                    words = cleaned_sentence.split()
                    
                    if len(words) == 0:
                        continue
                        
                    for word in words:
                        sentence_input.append(self.word_to_id.get(word,self.word_to_id["<unk>"]))
                    
                    review_input.append(torch.tensor(sentence_input,dtype=torch.long,requires_grad=False))
                    
                    review_labels.append(sentence_label) 
                
                
                
                instance = (review_input,review_labels,book_id,user_id,timestamp,rating)
                data[counter] = instance
                counter += 1
                
        print(counter)
        self.data = data
                


# train_dataset = SpoilerDataset('data/train_reviews.json','data/word_to_id.pickle','data/id_to_word.pickle')


# In[48]:


# train_dataloader = DataLoader(train_dataset)


# In[49]:


# for batch_idx, input_data in enumerate(train_dataloader):
    
#     print(input_data[0])


