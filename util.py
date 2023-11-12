"""
This file contains utility functions for the model.
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
# from pickle import dump, load
from sklearn.metrics import mean_absolute_error

def load_processed_data(file_path="data/processed/df_base_trainval_preprocessed.csv"):
    """
    This function loads the processed data.
    :param file_path: path to processed data
    :return: X, y
    """
    df = pd.read_csv(file_path, parse_dates=["Zeitstempel"])
    X, y = df.drop(["Zeitstempel", "MWh"], axis=1), df["MWh"]
    return X, y

def create_time_series_splits(X, n_splits=5):
    """
    This function creates time series splits for cross-validation.
    :param X: features
    :param n_splits: number of splits
    :return: splits
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    return splits

def train_preprocessor(X_train, preprocessor):
    """
    This function fits the preprocessor on the training data.
    :param X_train: training data
    :param preprocessor: preprocessor pipeline
    :return: fitted preprocessor
    """
    return preprocessor.fit(X_train)

def evaluate_model(y_true, y_pred):
    """
    This function calculates the evaluation metric.
    :param y_true: true values
    :param y_pred: predicted values
    :return: evaluation metric
    """
    return mean_absolute_error(y_true, y_pred)
