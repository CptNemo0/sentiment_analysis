# Sentiment analisys
This repository contains code in python that can classify a short text as a "positive" or "negative" depending on what is written. It is an AI based solution using LSTM (long short term memory) recurrent neural network architecture. 

## Files
- Train.ipynb - it is a base file that is used for training.
- my_callses.py - this file contains all important classes that were created during training and used in next 2 files
- TestAccMeasurement - in this file I measure accuracy of my RNN
- LiveTest.ipynb - this file is used to assess the sentiment of one standalone review (aka flex on my friends and family)

## Accuracy
First training - 84%
Second training - 86%

## Training
Text data is tokenized using tiktoken library, batched and fed into RNN. Model was not pretrained, nor it was finetuned. 

Before next training I used dataset that contained positive and negative words, and conducted pretraining. Than I trained on imdb dataset. I got better results after fewer epochs.

## Weights
Results of my first training are available under this URL: 

https://drive.google.com/file/d/1fPxhkbmFKWjIONkvGh-bJahJX-7ZnTNM/view?usp=sharing

SHA256: 57cb39b361c8280e20c5446ad19f3944d4d50775532f63d7805d8053918798e0

## Dataset 
https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
