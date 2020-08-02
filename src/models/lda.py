import pandas as pd
import re
import numpy as np
import nltk
import spacy
from spacy.lang.en import English
import en_core_web_sm
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
import pickle
from tqdm import tqdm
import pyLDAvis.gensim
import logging
import warnings

from data_processing.data_processing_helpers import DataProcessing
import data_processing.helpers as helpers
from data_processing.helpers import Config

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger('sarcasm_detector')


class LDAPreProcessing:
    """Perform preprocessing of data in order to perform LDA

    Preprocessing steps:
    - Tokenize the headlines
    - Ensure all words in a headline is in lowercase
    - Remove headlines with only a few words (<4 words)
    - Lemmatize the headlines
     """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    def __init__(self):
        self.df = DataProcessing().run()
        self.text_data = []
        self.dictionary = {}
        self.corpus = []
        en_core_web_sm.load()
        self.en_stop = set(nltk.corpus.stopwords.words('english'))
        self.parser = English()

    def tokenize(self, text):
        """this function is to tokenize the headline into a list of individual words"""
        lda_tokens = []
        # need to use parser for python to treat the list as words
        tokens = self.parser(text)
        for token in tokens:
            # to ignore any whitespaces in the headline, so that token list does not contain whitespaces
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)  # tokens (headlines) are already in lowercase
        return lda_tokens

    def get_lemma(self, word):
        """this function is to lemmatize the words in a headline into its root form"""
        # converts the word into root form from wordnet
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def prepare_text_for_lda(self, text):
        """To tokenize, remove headlines with only a few words (<4 words), lemmatize words in headlines"""
        tokens = self.tokenize(text)  # parse and tokenize the headline into a list of words
        tokens = [token for token in tokens if len(token) > 4]  # remove headlines with only length of 4 words or less
        tokens = [token for token in tokens if token not in self.en_stop]  # remove stopwords in the headline
        tokens = [self.get_lemma(token) for token in tokens]  # lemmatize the words in the headline
        return tokens

    def run(self):
        """Top-level method in class for preprocessing the data for LDA model"""
        logger.info('Starting data processing for LDA...')
        logger.info('Starting to tokenize dataset...')
        for i in tqdm(range(0, len(self.df.headline))):
            headline = self.df.headline[i]
            tokens = self.prepare_text_for_lda(headline)
            self.text_data.append(tokens)
        logger.info('Tokens created from dataset')
        logger.info('Starting to convert headlines into corpus and dictionary...')
        # Convert all headlines into a corpus of words, with each word as a token
        self.dictionary = corpora.Dictionary(self.text_data)
        # Convert each headline (a list of words) into the bag-of-words format. (Word ID, Count of word)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.text_data]
        logger.info('Corpus and dictionary created from dataset')
        pickle.dump(self.corpus, open('result/visualization/corpus.pkl', 'wb'))
        logger.info('Corpus saved as corpus.pkl in visualization folder')
        # takes a while to run the dictionary and corpus
        self.dictionary.save('result/visualization/dictionary.gensim')
        logger.info('Dictionary saved as dictionary.gensim in visualization folder')
        return self.dictionary, self.corpus


class LDA:
    """Perform LDA on the dataset and obtain the results in terms of number of topics and distribution of words
    in each topic

    Results:
    - Topics and distribution of words in each topic as a linear combination of coefficients (in results/logs)
    - Interactive visualization for each topic and distribution of words in each topic (in results/visualization)
     """
    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.dictionary, self.corpus = LDAPreProcessing().run()
        self.NUM_TOPICS = self.config.NUM_TOPICS

    def model(self):
        """this function is to tokenize the headline into a list of individual words"""
        ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.NUM_TOPICS,
                                                   id2word=self.dictionary, passes=15)
        ldamodel.save('result/visualization/{}.gensim'.format(str(self.NUM_TOPICS) + '_Topic'))
        logger.info('LDA Model saved as visualization/{}.gensim in visualization folder'.format(
                                                                                       str(self.NUM_TOPICS) + '_Topic'))
        logger.info('-----------------')
        logger.info('Results for LDA model with {} (top 5 words in each topic):'.format(str(self.NUM_TOPICS)))
        logger.info('-----------------')
        logger.info(ldamodel.print_topics(num_words=5))
        lda_display = pyLDAvis.gensim.prepare(ldamodel, self.corpus, self.dictionary, sort_topics=False)
        pyLDAvis.display(lda_display)
        pyLDAvis.save_html(lda_display, 'result/visualization/lda_{}.html'.format(str(self.NUM_TOPICS) + '_Topic'))
        logger.info(
            'LDA Model Visualization saved as visualization/lda_{}.html in visualization folder'.format(
                str(self.NUM_TOPICS) + '_Topic'))

    def run(self):
        """Top-level method in class for running the LDA to generate the HTML visualizations"""
        self.model()
