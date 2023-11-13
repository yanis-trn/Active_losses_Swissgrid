"""
This file contain objective function for hyperparameter tuning using optuna.
"""

from src.training import train_and_evaluate_model
import pandas as pd
import numpy as np
import warnings

def objective(trial, X, y, preprocessor, splits):
    params = {
        'alpha': trial.suggest_float('alpha', 0, 100),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
        'max_iter': trial.suggest_int('max_iter', 500, 5000)
    }

    mae_cv_list = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for train_index, test_index in splits:
            X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].reset_index(drop=True)

            evaluation_score = train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, params)
            mae_cv_list.append(evaluation_score)

    return np.mean(mae_cv_list)

