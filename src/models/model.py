import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_processing.data_processing_helpers import DataProcessing
from data_processing.helpers import Config
import data_processing.helpers as helpers
import logging

logger = logging.getLogger('sarcasm_detector')


class ModelPreProcessing:
    """Preprocess the dataset to be suitable for deep learning models RNN

    Steps in Preprocessing:
    - Split dataset into train/test dataset (default 90/10)
    - Tokenize, pad and truncate each headline (sequence of words)
     """

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)

        self.padded = []
        self.testing_padded = []
        self.df = DataProcessing().run()

        self.training_sentences = []
        self.training_labels = np.array([])

        self.testing_sentences = []
        self.testing_labels = np.array([])

    def train_test_split(self):
        """this function is to randomly split 10% of dataset to be testing dataset and 90% to be training dataset"""
        train_data, test_data = train_test_split(self.df[['headline', 'is_sarcastic']], test_size=0.1)
        self.training_sentences = list(train_data['headline'])
        self.training_labels = list(train_data['is_sarcastic'])

        self.testing_sentences = list(test_data['headline'])
        self.testing_labels = list(test_data['is_sarcastic'])

        self.training_labels = np.array(self.training_labels)
        self.testing_labels = np.array(self.testing_labels)

    def tokenize(self):
        """this function is to tokenize, pad the sequence of words in a headline"""
        tokenizer = Tokenizer(num_words=self.config.vocab_size, oov_token=self.config.oov_tok)
        tokenizer.fit_on_texts(self.training_sentences)

        sequences = tokenizer.texts_to_sequences(self.training_sentences)
        self.padded = pad_sequences(sequences, maxlen=self.config.max_length, truncating=self.config.trunc_type)

        testing_sequences = tokenizer.texts_to_sequences(self.testing_sentences)
        self.testing_padded = pad_sequences(testing_sequences, maxlen=self.config.max_length)

    def run(self):
        """Top-level method in class for running all other methods to preprocess the dataset"""
        logger.info('Data processing for RNN models...')
        self.train_test_split()
        self.tokenize()
        logger.info('Data processing for RNN models completed')
        return self.training_labels, self.testing_labels, self.padded, self.testing_padded


def plot_graphs(history, string, model):
    """this function is to use visualize the result from RNN model

    Figure 1: Accuracy & Validation Accuracy against number of epochs
    Figure 2: Loss & Validation Loss against number of epochs
    """
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.savefig('result/{0}/{1}.png'.format(model, string + ' on Epochs'))
    logger.info('Plot saved as {}.png in result/visualization folder'.format(string + ' on Epochs'))
    plt.show()


class GRU:
    """this class is to use GRU (RNN) model to predict whether a headline is sarcastic (1) or non-sarcastic (0),
     on a training dataset and validate on a testing dataset

    Model Definition with BiRNN (GRU):
    - L1 Lasso Regularization:      for feature selection
    - Dropout:                      for robustness of recurrent neural networks
    - Batch Normalization:          to stabilize and perhaps accelerate the learning process
     """

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.training_labels, self.testing_labels, self.padded, self.testing_padded = ModelPreProcessing().run()
        self.model = []
        self.history = []

    def compile_model(self):
        """Set up the model architecture and compile the deep learning RNN model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim,
                                      input_length=self.config.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.config.gru_units)),
            tf.keras.layers.Dense(self.config.dense_units,
                                  kernel_regularizer=tf.keras.regularizers.l1(self.config.regularizer_rate),
                                  activation=self.config.activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(self.config.regularizer_rate),
                                  activation='sigmoid')
        ])
        self.model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)

    def model_summary(self):
        """To view the model architecture via print model.summary"""
        print(self.model.summary())

    def run_model(self):
        """To compile and train the model"""
        self.compile_model()
        self.history = self.model.fit(self.padded, self.training_labels, epochs=self.config.num_epochs,
                                      batch_size=self.config.batch_size,
                                      validation_data=(self.testing_padded, self.testing_labels))
        return self.history

    def run(self):
        """Top-level method in class for running the GRU RNN model"""
        logger.info('Training GRU model ..')
        self.run_model()
        logger.info('GRU model training completed')
        plot_graphs(self.history, 'accuracy', self.config.model)
        plot_graphs(self.history, 'loss', self.config.model)


class LSTM:
    """this class is to use LSTM (RNN) model to predict whether a headline is sarcastic (1) or non-sarcastic (0),
     on a training dataset and validate on a testing dataset

    Model Definition with BiRNN (LSTM):
    - L1 Lasso Regularization:      for feature selection
    - Dropout:                      for robustness of recurrent neural networks
    - Batch Normalization:          to stabilize and perhaps accelerate the learning process
     """

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.training_labels, self.testing_labels, self.padded, self.testing_padded = ModelPreProcessing().run()
        self.model = []
        self.history = []

    def compile_model(self):
        """Set up the model architecture and compile the deep learning RNN model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim,
                                      input_length=self.config.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.config.lstm_units)),
            tf.keras.layers.Dense(self.config.dense_units, kernel_regularizer=tf.keras.regularizers.l1(
                self.config.regularizer_rate),
                                  activation=self.config.activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(self.config.regularizer_rate),
                                  activation='sigmoid')
        ])
        self.model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)

    def model_summary(self):
        """To view the model architecture via print model.summary"""
        print(self.model.summary())

    def run_model(self):
        """To compile and train the model"""
        self.compile_model()
        self.history = self.model.fit(self.padded, self.training_labels, epochs=self.config.num_epochs,
                                      batch_size=self.config.batch_size,
                                      validation_data=(self.testing_padded, self.testing_labels))
        return self.history

    def run(self):
        """Top-level method in class for running the LSTM RNN model"""
        logger.info('Training LSTM model ..')
        self.run_model()
        logger.info('LSTM model training completed')
        plot_graphs(self.history, 'accuracy', self.config.model)
        plot_graphs(self.history, 'loss', self.config.model)


class CNN:
    """this class is to use CNN model to predict whether a headline is sarcastic (1) or non-sarcastic (0),
     on a training dataset and validate on a testing dataset

    Model Definition with CNN:
    - L1 Lasso Regularization:      for feature selection
    - Dropout:                      for robustness of recurrent neural networks
    - Batch Normalization:          to stabilize and perhaps accelerate the learning process
     """

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.training_labels, self.testing_labels, self.padded, self.testing_padded = ModelPreProcessing().run()
        self.model = []
        self.history = []

    def compile_model(self):
        """Set up the model architecture and compile the deep learning RNN model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim,
                                      input_length=self.config.max_length),
            tf.keras.layers.Conv1D(self.config.filters, self.config.kernel_size, activation=self.config.activation),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(self.config.dense_units, kernel_regularizer=tf.keras.regularizers.l1(
                self.config.regularizer_rate),
                                  activation=self.config.activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(self.config.regularizer_rate),
                                  activation='sigmoid')
        ])
        self.model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)

    def model_summary(self):
        """To view the model architecture via print model.summary"""
        print(self.model.summary())

    def run_model(self):
        """To compile and train the model"""
        self.compile_model()
        self.history = self.model.fit(self.padded, self.training_labels, epochs=self.config.num_epochs,
                                      batch_size=self.config.batch_size,
                                      validation_data=(self.testing_padded, self.testing_labels))
        return self.history

    def run(self):
        """Top-level method in class for running the CNN Conv1D model"""
        logger.info('Training CNN model ..')
        self.run_model()
        logger.info('CNN model training completed')
        plot_graphs(self.history, 'accuracy', self.config.model)
        plot_graphs(self.history, 'loss', self.config.model)


class CNN_LSTM:
    """this class is to use CNN-LSTM model to predict whether a headline is sarcastic (1) or non-sarcastic (0),
     on a training dataset and validate on a testing dataset

    Model Definition with CNN-LSTM:
    - L1 Lasso Regularization:      for feature selection
    - Dropout:                      for robustness of recurrent neural networks
     """

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.training_labels, self.testing_labels, self.padded, self.testing_padded = ModelPreProcessing().run()
        self.model = []
        self.history = []

    def compile_model(self):
        """Set up the model architecture and compile the deep learning RNN model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim,
                                      input_length=self.config.max_length),
            tf.keras.layers.Conv1D(self.config.cnn_lstm_filters, self.config.cnn_lstm_kernel_size,
                                   activation=self.config.activation),
            tf.keras.layers.MaxPooling1D(2, padding="same"),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.config.cnn_lstm_units)),

            tf.keras.layers.Dense(self.config.dense_units, kernel_regularizer=tf.keras.regularizers.l1(
                self.config.regularizer_rate),
                                  activation=self.config.activation),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(self.config.regularizer_rate),
                                  activation='sigmoid')
        ])
        self.model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)

    def model_summary(self):
        """To view the model architecture via print model.summary"""
        print(self.model.summary())

    def run_model(self):
        """To compile and train the model"""
        self.compile_model()
        self.history = self.model.fit(self.padded, self.training_labels, epochs=self.config.num_epochs,
                                      batch_size=self.config.batch_size,
                                      validation_data=(self.testing_padded, self.testing_labels))
        return self.history

    def run(self):
        """Top-level method in class for running the CNN-LSTM model"""
        logger.info('Training CNN-LSTM model ..')
        self.run_model()
        logger.info('CNN-LSTM model training completed')
        plot_graphs(self.history, 'accuracy', self.config.model)
        plot_graphs(self.history, 'loss', self.config.model)
