# Sarcasm Detector for News Headline
kaggle link: https://www.kaggle.com/jinghuiwong/lstm-cnn-with-tensorflow-lda-topic-modelling


# Requirements
* Tensorflow 2.0
* pandas
* numpy
* scikit-learn 0.21.3
* matplotlib
* nltk 3.4.5
* spacy 2.2.3
* pyLDAvis 2.1.2
* gensim 3.8.1


# Objective 

Given the rise of fake news on social media as well as on news outlets, it is important for people to identify sarcastic news reports from legitimate news reports. This will allow people to know when to take the news at face value and not spread false rumours. Given a news headline, predict whether a news headline is a sarcastic remark or not. In this repository, the following unsupervised and supervised learning methods are used: Latent Dirichlet Allocation for topic extraction, Deep Learning models (RNN with GRU, RNN with LSTM, CNN and finally a neural network architecture that combines both CNN and RNN). To ensure the robustness and to stabilize the performance of deep learning methods, techniques such as batch normalization, dropout layers and regularization are also used. 

# Overview
This repository begins with the data processing (lowercase, remove punctuation, etc) and carry out an exploratory data analysis (such as unique word count in each headline, total word length for headline, use of numbers/statistics in headline, etc). To understand the topics of headline, Latent Dirichlet Allocation (LDA) is also used. Before running the LDA model, the necessary steps of tokenization, stop word removal, removal of headlines with a short length, lemmatization, stemming are carried out. For the task of predicting sarcastic headlines, various Recurrent Neural Network (RNN) architectures are used. For example, RNN with Gated Recurrent Unit or RNN with Long Short Term Memory, Convolutional Neural Networks using only 1-Dimensional Convolutional Layer (Conv1D), and finally a combination of CNN and RNN. Based on the performance on the test set, the combined architecture of CNN-RNN performs the best, at 86% accuracy. Further fine-tuning of the model will help to improve this performance. 

# Context
Past studies in Sarcasm Detection mostly make use of Twitter datasets collected using hashtag based supervision but such datasets are noisy in terms of labels and language. Furthermore, many tweets are replies to other tweets and detecting sarcasm in these requires the availability of contextual tweets.

To overcome the limitations related to noise in Twitter datasets, this News Headlines dataset for Sarcasm Detection is collected from two news website. TheOnion aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from HuffPost.

This new dataset has following advantages over the existing Twitter datasets:

- Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the chance of finding pre-trained embeddings.

- Furthermore, since the sole purpose of TheOnion is to publish sarcastic news, we get high-quality labels with much less noise as compared to Twitter datasets.

- Unlike tweets which are replies to other tweets, the news headlines we obtained are self-contained. This would help us in teasing apart the real sarcastic elements.

Each record consists of three attributes:

`is_sarcastic` : 1 if the record is sarcastic otherwise 0

`headline`     : the headline of the news article

`article_link`: link to the original news article. Useful in collecting supplementary data

# Using the Package

## Makefile

To run the package using `Makefile`, simply run the following commands below:

```
# To install the required modules and setup.py
make setup

# To create visualization plots. Plots are saved in results folder
make visualize

# To perform LDA. Interactive visualization is saved in results folder
make lda

# To run the RNN deep learning models
make run-model

# To run pytest on the package
make test
```

## Using Shellscript

To run the package as shellscript, simply click on the file `run.sh`

## Using Click

To run the package on cmd, simply run the following commands below:

```
python setup.py sdist
pip install dist/sarcasm_detector_news_headline-0.1.0.tar.gz

# To install the required modules and setup.py
sarcasm_detector_news_headline visualize

# To perform LDA. Interactive visualization is saved in results folder
sarcasm_detector_news_headline lda

# To run the RNN deep learning models
sarcasm_detector_news_headline run-model
```

# Parameters for RNN Models

The parameters for the RNN Models is found in `config.yaml`. Edit the parameter `model` to select which model to run.

## Convolutional (Conv1D) Model Parameters
* num_epochs
* batch_size
* dense_units
* dropout_rate
* regularizer_rate
* activation
* loss
* optimizer
* filters
* kernel_size

## Gated Recurrent Unit (GRU) Model Parameters
* num_epochs
* batch_size
* dense_units
* dropout_rate
* regularizer_rate
* activation
* loss
* optimizer
* gru_units

## Long Short Term Memory (LSTM) Model Parameters
* num_epochs
* batch_size
* dense_units
* dropout_rate
* regularizer_rate
* activation
* loss
* optimizer
* lstm_units

## Long Short Term Memory (LSTM) Model Parameters
* num_epochs
* batch_size
* dense_units
* dropout_rate
* regularizer_rate
* activation
* loss
* optimizer
* cnn_lstm_filters
* cnn_lstm_kernel_size
* cnn_lstm_units

# References:
[1] Using LDA for topic extraction: https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc

[2] Guidelines for the use of LDA: https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc

[3] Topic modelling in python with nltk and gensim: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

# Kaggle Link:
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/kernels
