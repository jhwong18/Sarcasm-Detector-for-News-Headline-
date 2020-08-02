import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import logging

from data_processing.data_processing_helpers import DataProcessing

logger = logging.getLogger('sarcasm_detector')

class Visualization:
    """Visualize the processed dataset and output the plots into the result/visualization folder

    Visualizations:
    - Compare Frequencies of Sarcastic vs Non-Sarcastic Headlines
    - Compare Proportion of Sarcastic vs Non-Sarcastic Headlines
    - Analyse the distribution of Headlines length over entire dataset
    - Analyse the distribution of Headlines length over sarcastic dataset
    - Analyse the distribution of Headlines length over non-sarcastic dataset
     """
    def __init__(self):
        self.df = DataProcessing().run()
        self.df_viz = self.df

    def df_groupby_col(self, col='is_sarcastic'):
        """Groupby dataset based on a specified column"""
        self.df_viz = self.df.groupby(col).count()
        if col == 'is_sarcastic':
            self.df_viz.index = ['Non-sarcastic', 'Sarcastic']
        return self.df_viz

    def plot_bar(self, x, y, xlabel, ylabel, title, **other_params):
        """Plot a histogram figure with optional parameters (vertical lines, coloured bars, etc)"""
        plt.figure(figsize=(15, 10))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(fontsize=10)
        plt.title(title)
        bar_graph = plt.bar(x, y)
        if 'set_color_ind' in other_params.keys():
            bar_graph[other_params['set_color_ind']].set_color('r')
        if 'formatter' in other_params.keys():
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        if 'vline' in other_params.keys():
            plt.axvline(other_params['vline'], color='k', linestyle='dashed', linewidth=1)
        plt.savefig('result/visualization/{}.png'.format(ylabel+' on '+xlabel))
        logger.info('Plot saved as {}.png in results/visualization folder'.format(ylabel + ' on ' + xlabel))
        plt.show()

    def run(self):
        """Top-level method in class for running all other methods to generate the visualizations"""
        # Figure 1: Compare Frequencies of Sarcastic vs Non-Sarcastic Headlines
        df_tmp = self.df_groupby_col()
        self.plot_bar(df_tmp.index, df_tmp.headline_count,
                      xlabel='Sarcastic and Non-sarcastic Headlines',
                      ylabel='Frequencies of headlines',
                      title='Frequencies of Sarcastic vs Non-sarcastic headlines',
                      set_color_ind=1)

        # Figure 2: Compare Proportion of Sarcastic vs Non-Sarcastic Headlines
        self.plot_bar(df_tmp.index, df_tmp.headline_count / df_tmp.headline_count.sum(),
                      xlabel='Sarcastic and Non-sarcastic Headlines',
                      ylabel='Proportion of headlines',
                      title='Proportion of Sarcastic vs Non-sarcastic headlines',
                      set_color_ind=1, formatter=True)

        # Figure 3: Distribution of Headlines length over entire dataset
        df_tmp = self.df_groupby_col(col='headline_count')

        self.plot_bar(df_tmp.index, df_tmp.headline,
                      xlabel='Different lengths of headline',
                      ylabel='Frequencies of headline',
                      title='Distribution of headline length for entire dataset',
                      set_color_ind=8, vline=self.df.headline_count.mean())

        # Figure 4: Distribution of Headlines length over sarcastic dataset
        df_tmp_sarcastic_ = self.df[self.df.is_sarcastic == 1]
        df_tmp_sarcastic = df_tmp_sarcastic_.groupby('headline_count').count()

        self.plot_bar(df_tmp_sarcastic.index, df_tmp_sarcastic.headline,
                      xlabel='Different lengths of sarcastic headline',
                      ylabel='Frequencies of sarcastic headline',
                      title='Distribution of headline length for sarcastic dataset',
                      set_color_ind=7, vline=self.df.headline_count.mean())

        # Figure 5: Distribution of Headlines length over non-sarcastic dataset
        df_tmp_non_sarcastic_ = self.df[self.df.is_sarcastic == 0]
        df_tmp_non_sarcastic = df_tmp_non_sarcastic_.groupby('headline_count').count()

        self.plot_bar(df_tmp_non_sarcastic.index, df_tmp_non_sarcastic.headline,
                      xlabel='Different lengths of non-sarcastic headline',
                      ylabel='Frequencies of non-sarcastic headline',
                      title='Distribution of headline length for non-sarcastic dataset',
                      set_color_ind=8, vline=self.df.headline_count.mean())
