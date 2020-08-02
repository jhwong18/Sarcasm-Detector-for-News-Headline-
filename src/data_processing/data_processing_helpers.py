import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
import logging
import string
import re

from data_processing.helpers import Config
import data_processing.helpers as helpers

logger = logging.getLogger('sarcasm_detector')


class DataLoad:
    """Load the JSON dataset, convert into pandas dataframe and check for null values in each column"""
    def __init__(self, data_path="data/Sarcasm_Headlines_Dataset.json"):
        self.data_path = data_path
        self.df = None

    def load_jsondata(self):
        """Load JSON dataset and convert into pandas dataframe"""
        self.df = pd.read_json(self.data_path, lines=True)
        logger.info('{} dataset loaded'.format(self.data_path))

    def filter_columns(self):
        """Remove columns in dataset"""
        self.df = self.df[['headline', 'is_sarcastic']]

    def check_null_for_sarcasm(self):
        """Check for null values in target variable: is_sarcastic"""
        logger.info('{} columns with valid target values'.format(len(self.df.is_sarcastic)))
        logger.info('{} columns with empty target values'.format(len(self.df.is_sarcastic) -
                                                                 len(self.df.is_sarcastic.isnull())))

    def check_null_for_headlines(self):
        """Check for null values in feature variable: headline"""
        logger.info('{} columns with valid headlines'.format(len(self.df.headline)))
        logger.info('{} columns with empty headlines'.format(len(self.df.headline) -
                                                             len(self.df.headline.isnull())))

    def run(self):
        """Top-level method in class for running all other methods in the class"""
        self.load_jsondata()
        self.filter_columns()
        self.check_null_for_headlines()
        self.check_null_for_sarcasm()
        return self.df


class DataProcessing:
    """Perform preprocessing of data in order to perform the subsequent analysis and modelling

    Preprocessing steps:
    - Ensure all words in a headline is in lowercase
    - Remove punctuation in headlines
    - Create a new column to calculate the number of words in a headline
    - Create a new column to calculate the number of unique words in a headline
    - Create a new column to determine whether a headline contains numbers/digits
     """
    def __init__(self):
        self.df = DataLoad().run()

    def convert_headline_lowercase(self):
        """Ensure all words in a headline is in lowercase"""
        self.df['headline'] = self.df['headline'].apply(lambda x: x.lower())

    def remove_headline_punctuation(self):
        """Remove punctuation in headlines"""
        self.df['headline'] = self.df['headline'].apply(lambda x: ' '.join(word.strip(string.punctuation)
                                                                           for word in x.split()))

    def create_headline_count(self):
        """Create a new column to calculate the number of words in a headline """
        self.df['headline_count'] = self.df['headline'].apply(lambda x: len(list(x.split())))

    def create_headline_unique_count(self):
        """Create a new column to calculate the number of unique words in a headline"""
        self.df['headline_unique_word_count'] = self.df['headline'].apply(lambda x: len(set(x.split())))

    def create_headline_digit(self):
        """Create a new column to determine whether a headline contains numbers/digits"""
        self.df['headline_has_digits'] = self.df['headline'].apply(lambda x: bool(re.search(r'\d', x)))

    def run(self):
        """Top-level method in class for running all other methods in the class"""
        logger.info('Starting data processing...')
        self.convert_headline_lowercase()
        self.remove_headline_punctuation()
        self.create_headline_count()
        self.create_headline_unique_count()
        self.create_headline_digit()
        logger.info('Data processing completed')
        return self.df
