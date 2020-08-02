import logging
import yaml
import json
import sys
import os

logger = logging.getLogger('sarcasm_detector')

class Config:
    """Loads parameters from config.yaml into global object

    Args:
            config_path (str): path to config.yaml
    Returns:
        dictionary (obj): a dictionary containing all parameter values from config.yaml
    """

    def __init__(self, path_to_config):

        self.path_to_config = path_to_config
        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup


def setup_logging():
    """Configure logging object to track parameter settings, training, and evaluation.

    Returns:
        logger (obj): Logging object
    """

    logger = logging.getLogger('sarcasm_detector')
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger


def make_dirs():
    """Create directories for storing data in repo if they don't already exist"""
    paths = ['result', 'result/GRU', 'result/logs', 'result/CNN', 'result/LSTM', 'result/CNN_LSTM',
             'result/visualization']
    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)