"""
This file contains the training code for the model.
"""

from sklearn.linear_model import ElasticNet
from util import train_preprocessor, evaluate_model
from visualization import visualize_predictions
from pickle import dump, load
import pandas as pd

def train_final(X, y, preprocessor, params):
    """
    This function trains the final model for later use on other data.
    :param X: features
    :param y: target
    :param preprocessor: preprocessor pipeline
    :param params: hyperparameters
    :return: fitted model
    """
    fitted_preprocessor = train_preprocessor(X, preprocessor)

    X = pd.DataFrame(
        fitted_preprocessor.transform(X),
        columns=fitted_preprocessor.get_feature_names_out(),
    )

    # Train model
    model = ElasticNet(**params)
    model.fit(X, y)

    with open("saved_models/final_model.pickle", 'wb') as file:
        dump(model, file)

    return model


def train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, params):
    """
    This function trains and evaluates the model. Function used for hyperparameter tuning.
    :param X_train: training data
    :param X_test: test data
    :param y_train: training target
    :param y_test: test target
    :param preprocessor: preprocessor pipeline
    :param params: hyperparameters
    :return: evaluation metric
    """
    fitted_preprocessor = train_preprocessor(X_train, preprocessor)

    X_train = pd.DataFrame(
        fitted_preprocessor.transform(X_train),
        columns=fitted_preprocessor.get_feature_names_out(),
    )

    X_test = pd.DataFrame(
        fitted_preprocessor.transform(X_test),
        columns=fitted_preprocessor.get_feature_names_out(),
    )

    # Train model
    model = ElasticNet(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    # Calculate MAE for each fol
    evaluation_score = evaluate_model(y_test, y_pred)

    return evaluation_score

