#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import json
import pickle
import string
from dataset import SpoilerDataset
from sklearn.metrics import confusion_matrix
# from tqdm import tqdm
from utils import *
import time
from Loss_functions import *
import warnings
warnings.filterwarnings('ignore')
# %load_ext autoreload
# %autoreload 2


# In[2]:


# SUFFIX_ALL = ""
EPOCHS = (50,60)
WORD_EMBEDDING_DIM = 300
HIDDEN_DIM = 50
WORD_FEATURES_DIM = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
# DEVICE = torch.device("cpu")
LEARNING_RATE = 0.001
ACCUMULATE_GRAD_STEPS = 128
LOAD_PROCESSED_DATA = True
SAVE_MODEL = True
SAVE_LOG_TO_FILE = True
NEGATIVE_CLASS_WEIGHT = 0.5
LOAD_MODEL = True
MODEL_VERSION = 30
TRAIN_TIME_THRESHOLD = 10000
OUT_FILE = "output_V8.txt"


# In[3]:


class SpoilerNet(nn.Module):
    def __init__(self, train_dataset, word_emb_dim, features_dim, hidden_dim, word_vocab_size, num_gru_layers, bidirectional):
        super(SpoilerNet, self).__init__()
        self.device = DEVICE
        self.dataset = train_dataset
        pretrained_embeddings = self.dataset.load_pretrained_embeddings(word_emb_dim)
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_embeddings.to(self.device), freeze=True)
        # adding word features
        word_emb_dim += features_dim
        self.word_gru = nn.GRU(input_size=word_emb_dim, hidden_size=hidden_dim, 
                               num_layers=num_gru_layers, bidirectional=bidirectional ,batch_first=True)
        self.linear = nn.Linear(in_features=2*hidden_dim,out_features=hidden_dim)
        self.attention = nn.Linear(in_features=hidden_dim,out_features=1,bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sentence_gru = nn.GRU(input_size=2*hidden_dim,hidden_size=hidden_dim,num_layers=num_gru_layers,
                                   bidirectional=bidirectional,batch_first=True)
        self.output_layer = nn.Linear(in_features=2*hidden_dim,out_features=2)
        self.book_bias = nn.Parameter(torch.torch.zeros(self.dataset.get_num_books()), requires_grad=True)
        self.user_bias = nn.Parameter(torch.torch.zeros(self.dataset.get_num_users()), requires_grad=True)
#         self.logsoftmax_out = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        
    def forward(self,review, book_ind, user_ind):
        vectorized_sentences = []
        for sentence in review:
            sentence = sentence.to(self.device)
            embedded_sentence = self.word_embedding(sentence).to(self.device)
            word_features = self.dataset.get_tf_idf_features_tensor(sentence, book_ind).to(self.device)
            embedded_sentence = torch.cat([embedded_sentence.float(), word_features.float()],dim=2)
            word_hidden_state, _ = self.word_gru(embedded_sentence)
            mu = self.tanh(self.linear(word_hidden_state))
            alpha_weights = self.softmax(self.attention(mu))
            attended_vector = (alpha_weights * word_hidden_state).sum(dim=1)
            vectorized_sentences.append(attended_vector)
        stacked_vectorized_sentences = torch.stack(vectorized_sentences,dim=1)
        sentence_hidden_state , _ = self.sentence_gru(stacked_vectorized_sentences)
        output = self.output_layer(sentence_hidden_state).view(len(review),-1)
        output += self.book_bias[book_ind] + self.book_bias[user_ind]
#         scores = self.logsoftmax_out(output)
        probs = self.softmax(output)
        return probs, output

    def predict(self,dataloader):
        self.eval()
        y_true = []
        y_pred = []
        y_probs = []
        softmax = nn.Softmax()
        for batch_idx, input_data in enumerate(dataloader):
#         for batch_idx, input_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            review = input_data[0]
            labels = torch.tensor(input_data[1]).to(self.device)
            book_id = input_data[2]
            user_id = input_data[3]
            scores, ouput = self.forward(review, book_id, user_id)
            probs = softmax(scores)
            positive_probs = probs[:,1]
            _, predicted = torch.max(probs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()
            y_probs += positive_probs.tolist()
        metrics = calc_metrics(y_true, y_pred, y_probs)
        metrics_str = json.dumps(metrics, indent=4)
        if SAVE_LOG_TO_FILE:
            write_to_log(metrics)
        print(metrics_str)
        self.train()
        return y_true, y_pred, metrics


def calc_metrics(y_true, y_pred, y_probs):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true,y_probs)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mat = np.array([tn, fp, fn, tp])
    mat = mat/mat.sum()
    matrics = np.concatenate([np.array([auc, precision, recall, f1, accuracy]), mat]).round(2)
    metrics = {"AUC":matrics[0], "precision": matrics[1], "recall":matrics[2], "f1":matrics[3], "accuracy":matrics[4],
               "TN":matrics[5], "FP":matrics[6], "FN":matrics[7], "TP":matrics[8],
               "Neg-Pos (pred)": f"[{round((len(y_pred)-sum(y_pred))/len(y_pred),2)}, {round(sum(y_pred)/len(y_pred),2)}]",
               "Neg-Pos (true)": f"[{round((len(y_true)-sum(y_true))/len(y_true),2)}, {round(sum(y_true)/len(y_true),2)}]",
               "Sum": len(y_true)}
    return metrics


# %load_ext autoreload
# %autoreload 2


# In[4]:


def write_to_log(obj):
    if not SAVE_LOG_TO_FILE:
        return
    with open(OUT_FILE, "a") as file:
        file.write(str(obj))
        file.write("\n")


def train():
    write_to_log("--"*50)
    write_to_log("--"*50)
    write_to_log("New Training \t"*5)
    train_dataset = SpoilerDataset(filename="train.pickle", load=LOAD_PROCESSED_DATA) # using defaults
    train_dataloader = DataLoader(train_dataset)
    
    valid_small_dataset = SpoilerDataset(filename="valid.pickle", load=LOAD_PROCESSED_DATA)
    valid_small_dataloader = DataLoader(valid_small_dataset)
    
    model = SpoilerNet(train_dataset, WORD_EMBEDDING_DIM, WORD_FEATURES_DIM, HIDDEN_DIM,len(train_dataset.word_to_id), 2, True)
    
    if LOAD_MODEL:
        load_model(model,MODEL_VERSION)
        model.train()
        
    model.zero_grad()
    model.to(DEVICE)
    
    # reduction='sum' because we're using 1 sample batch
#     criterion = nn.NLLLoss(weight=torch.tensor([NEGATIVE_CLASS_WEIGHT, 1]), reduction='sum').to(DEVICE)
#     criterion = DiceLoss(smooth=0.1, square=True)
    criterion = DSC(smooth=0.1)
#     criterion = FocalLoss(1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    beg = time.time()
    loss_train_list = []
    for epoch in range(*EPOCHS):
        loss_train_total = 0
        # each batch is a single review (many sentences)
#         for batch_idx, input_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        for batch_idx, input_data in enumerate(train_dataloader):
            review = input_data[0]
            labels = torch.tensor(input_data[1]).to(model.device)
            book_id = input_data[2]
            user_id = input_data[3]
            
            scores, output = model(review, book_id, user_id)
            loss = criterion(scores, labels)
            loss = loss/ACCUMULATE_GRAD_STEPS
            loss.backward()
            
            if batch_idx % ACCUMULATE_GRAD_STEPS == 0:
                optimizer.step()
                model.zero_grad()

            loss_train_total += loss.item()
#             break
            if time.time()-beg > TRAIN_TIME_THRESHOLD:
                _, _, _ = model.predict(valid_small_dataloader)
                write_to_log(str(round(time.time()-beg,2)) + "\t" + str(batch_idx))
                write_to_log("\n")
                beg = time.time()

        if SAVE_MODEL:
            save_model(model, epoch+1)

        loss_train_total = loss_train_total / len(train_dataset)
        loss_train_list.append(float(loss_train_total))

        print("Epoch {} Completed,\tTrain Loss: {}".format(epoch + 1, np.mean(loss_train_list[-batch_idx:])))
        
        if SAVE_LOG_TO_FILE:
            write_to_log("Epoch {} Completed,\tTrain Loss: {}".format(epoch + 1, np.mean(loss_train_list[-batch_idx:])))
            
    return model, loss_train_list


# In[5]:


model, _ = train()


# In[6]:


# train_dataset = SpoilerDataset(filename="train.pickle", load=LOAD_PROCESSED_DATA) # using defaults
# train_dataloader = DataLoader(train_dataset)
# model = SpoilerNet(train_dataset, WORD_EMBEDDING_DIM, WORD_FEATURES_DIM, HIDDEN_DIM,len(train_dataset.word_to_id), 2, True)
# load_model(model,MODEL_VERSION)
# model.to(DEVICE)

write_to_log("\n\nStart Testing")


# In[7]:


valid_dataset = SpoilerDataset(filename="valid.pickle", load=LOAD_PROCESSED_DATA)
valid_dataloader = DataLoader(valid_dataset)


# In[8]:


_, _, _ = model.predict(valid_dataloader)


# In[9]:


test_dataset = SpoilerDataset(filename="test.pickle", load=LOAD_PROCESSED_DATA)
test_dataloader = DataLoader(test_dataset)


# In[7]:


_, _, _ = model.predict(test_dataloader)


# In[ ]:


train_dataset = SpoilerDataset(filename="train.pickle", load=LOAD_PROCESSED_DATA)
train_dataloader = DataLoader(train_dataset)


# In[ ]:


_, _, _ = model.predict(train_dataloader)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




