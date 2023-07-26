# Sentiment analisys
This repository contains code in python that can classify a short text as a "positive" or "negative" depending on what is written. It is an AI based solution using LSTM (long short term memory) recurrent neural network architecture. 

## Files
- Train.ipynb - it is a base file that is used for training.
- my_callses.py - this file contains all important classes that were created during training and used in next 2 files
- TestAccMeasurement - in this file I measure accuracy of my RNN
- LiveTest.ipynb - this file is used to assess the sentiment of one standalone review (aka flex on my friends and family)

## Accuracy
First training - 84%

## Training
Text data is tokenized using tiktoken library, batched and fed into RNN. Model was not pretrained, nor it was finetuned. 

## Weights
Results of my first training are available under this URL