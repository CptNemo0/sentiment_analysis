import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

class ReviewsDataset(Dataset):
    def __init__(self, full_dataset, msl):
        self.sentiment = full_dataset[:, 0]
        self.data = full_dataset[:, 1]
        self.msl = msl
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
    def __len__(self):
        assert(len(self.sentiment) == len(self.data))
        return len(self.sentiment)
    
    def encode_data(self, selected_data):        
        encoded_data = []
        for sd in selected_data:
            encoded_data.append(self.encoder.encode(sd))
        
        truncated_data = []
        for e in encoded_data:
            if len(e) < self.msl:
                to_add = self.msl - len(e)
                for i in range(to_add):
                    e.append(220)
                truncated_data.append(e)
            elif len(e) > self.msl:
                truncated_data.append(e[:self.msl])
            elif len(e) == self.msl:
                truncated_data.append(e)
        return truncated_data
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        selected_data = self.data[idx]
        encoded_data = self.encode_data(selected_data)
        inputs = np.array(encoded_data).astype('float32')
        inputs = torch.tensor(inputs)

        targets = self.sentiment[idx]
        targets = np.array(targets).astype('int32')
        targets = torch.tensor(targets)
        targets = targets.to(torch.int64)
        one_hot = F.one_hot(targets, num_classes=2)
        
        sample = {'inputs': inputs, 'targets': one_hot}
        
        return sample

class MyLoader():
    def __init__(self, dataset, batch_size, generator):
        self.batch_size = batch_size
        self.dataset = dataset
        self.generator = generator
        
    def get_batch(self):
        ix = torch.randint(0, len(self.dataset), (self.batch_size,), generator = self.generator)
        return self.dataset.__getitem__(ix)
    
class Net(torch.nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, hidden_size, labels):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm      = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.lin       = nn.Linear(hidden_size, labels)
     
    def forward(self, data): #torch.Size([70, 32])
        embedding_output = self.embedding(data)  
        output, (h_n, c_n) = self.lstm(embedding_output)
        h_n.squeeze_(0) #torch.Size([1, 32, 70]) -- > torch.Size([32, 70])   
        output = self.lin(h_n)
        return output