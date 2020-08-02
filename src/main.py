from data_processing.data_processing_helpers import DataLoad, DataProcessing
import data_processing.helpers as helpers
from data_processing.helpers import Config

from visualization.visualization import Visualization

from models.lda import LDAPreProcessing, LDA
from models.model import ModelPreProcessing, GRU, LSTM, CNN, CNN_LSTM

import click
import logging


@click.group()
def cli():
    helpers.make_dirs()
    logger = helpers.setup_logging()
    # add logging FileHandler based on ID
    hdlr = logging.FileHandler('result/logs/sarcasm_detector.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    pass


@cli.command()
def run_model():
    """Run the following deep learning models based on specified parameters in config.yaml"""
    config = Config('config.yaml')
    if config.model == 'GRU':
        model = GRU()
    elif config.model == 'LSTM':
        model = LSTM()
    elif config.model == 'CNN':
        model = CNN()
    else:
        model = CNN_LSTM()
    model.run()


@cli.command()
def visualize():
    """Visualize the processed dataset and output the plots into the result/visualization folder"""
    Visualization().run()


@cli.command()
def lda():
    """Perform LDA on the dataset and obtain the results in terms of number of topics and distribution of words"""
    LDA().run()
