"""
This file contains utility functions for the model.
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from pickle import dump, load
import optuna

def load_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function loads the processed data.
    :param file_path: path to processed data
    :return: X, y
    """
    df['Zeitstempel'] = pd.to_datetime(df['Zeitstempel'])
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

def analyze_optuna_study(study):
    # Get pruned and complete trials
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    # Access the best trial
    best_trial = study.best_trial

    # Access the minimal MAE obtained during the training and the best hyperparameters
    min_mae = best_trial.value
    best_params = study.best_params

    # Display the study statistics
    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print(f"Minimal MAE: {min_mae}")
    print(best_params)

    # Save trials DataFrame to CSV
    study.trials_dataframe().to_csv("saved_models/trials_tuning_optuna.csv", index=False)

    # Save best hyperparameters to a pickle file
    with open("saved_models/best_model_hyperparams.pickle", 'wb') as file:
        dump(study.best_params, file)