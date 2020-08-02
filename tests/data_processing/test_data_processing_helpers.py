import os
import sys
import pytest
import pandas as pd

from data_processing.data_processing_helpers import DataLoad, DataProcessing


def test_load_jsondata():
    data = DataLoad()
    data.load_jsondata()
    assert isinstance(data.df, pd.DataFrame)


def test_filter_columns():
    data = DataLoad()
    data.load_jsondata()
    data.filter_columns()
    assert data.df[['headline', 'is_sarcastic']].shape[1] == 2