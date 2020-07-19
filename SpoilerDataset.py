


import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
import json
import pickle
import string



# In[5]:


# instance attributes = (review_input_tensor,review_labels_tensor,bookd_id,user_id,timestamp,rating)


# In[46]:

# In[15]:


# instance attributes = (words_review_tensor,summary_review_tensor,label,review_date,movie_id,user_id)


# In[6]:



class SpoilerDataset(Dataset):
    def __init__(self,dataset_path,word_to_id_path,id_to_word_path):
        super().__init__()
        self.file = dataset_path
        
        word_to_id_file= open(word_to_id_path, 'rb')
        self.word_to_id = pickle.load(word_to_id_file)
        word_to_id_file.close()
        
        id_to_word_file = open(id_to_word_path, 'rb')
        self.id_to_word = pickle.load(id_to_word_file)
        id_to_word_file.close()
        
        self.data = self.prepare_dataset()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
    
    
    def prepare_dataset(self):
        data = {}
        counter = 0
        with open(self.file) as json_file:
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

                    for word in words:
                        sentence_input.append(self.word_to_id.get(word,self.word_to_id["<unk>"]))
                    
                    review_input.append(torch.tensor(sentence_input,dtype=torch.long,requires_grad=False))
                    
                    review_labels.append(sentence_label) 


                    instance = (review_input,review_labels,book_id,user_id,timestamp,rating)

                    data[counter] = instance
                    counter += 1
                   
            return data
                


# train_dataset = SpoilerDataset('data/train_reviews.json','data/word_to_id.pickle','data/id_to_word.pickle')


# In[48]:


# train_dataloader = DataLoader(train_dataset)


# In[49]:


# for batch_idx, input_data in enumerate(train_dataloader):
    
#     print(input_data[0])


