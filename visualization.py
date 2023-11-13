"""
This file contains functions for visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from plotnine import *
from IPython.display import clear_output, display
import warnings


def visualize_predictions(y_true, y_pred):
    """
    Visualize the true vs predicted values.
    :param y_true: true target values
    :param y_pred: predicted target values
    :return: None
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.show()

def visualize_splits(splits, df, y):
    """
    This function visualizes the splits.
    :param splits: splits
    :param df: dataframe
    :param y: target variable
    :return: None
    """

    for split_idx, (train_index, test_index) in enumerate(splits):
        train_dates = df["Zeitstempel"].iloc[train_index]
        test_dates = df["Zeitstempel"].iloc[test_index]

        y_train, y_test = pd.DataFrame(
            {
                "Zeitstempel": train_dates.reset_index(drop=True),
                "MWh": y.iloc[train_index].reset_index(drop=True),
                "Period": "Training",
            }
        ), pd.DataFrame(
            {
                "Zeitstempel": test_dates.reset_index(drop=True),
                "MWh": y.iloc[test_index].reset_index(drop=True),
                "Period": "Testing",
            }
        )

        with warnings.catch_warnings():  # sanest pandas user
            warnings.simplefilter("ignore")

            p = (
                ggplot(
                    data=pd.concat([y_train, y_test], axis=0),
                    mapping=aes(x="Zeitstempel", y="MWh", colour="Period"),
                )
                + labs(title=f"Split {split_idx + 1}")
                + geom_line()
                + scale_colour_manual(values=["firebrick", "dodgerblue"])
                + theme_light()
                + theme(figure_size=[15, 2])
            )

            # Save the subplot to the corresponding axis
            p.save(filename=f"plots/split_{split_idx + 1}.png", format="png", verbose=False)
            # p.draw()
            # display(p)


def visualize_model_predictions(splits, X, y, preprocessor, params):
    """
    This function visualizes the model predictions.
    :param splits: splits
    :param X: features
    :param y: target
    :param preprocessor: preprocessor pipeline
    :param params: hyperparameters
    :return: None
    """
    with warnings.catch_warnings():  # sanest pandas user
        warnings.simplefilter("ignore")

        for split_idx, (train_index, test_index) in enumerate(splits):
            X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].reset_index(drop=True)
            
            # Fit preprocessor on each training split
            fitted_preprocessor = preprocessor.fit(X_train)
            
            X_train = pd.DataFrame(
                fitted_preprocessor.transform(X_train),
                columns=fitted_preprocessor.get_feature_names_out(),
            )

            X_test = pd.DataFrame(
                fitted_preprocessor.transform(X_test),
                columns=fitted_preprocessor.get_feature_names_out(),
            )
            
            model = ElasticNet(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            x_axis = range(len(y_test))

            data = {"y_test": y_test, "y_pred": y_pred}
            df = pd.DataFrame(data)

            # Set the style of seaborn
            sns.set(style="whitegrid")

            # Create a line plot
            plt.figure(figsize=(14, 6))  # Set the size of the plot
            sns.lineplot(data=df, dashes=False) 

            # Add labels and title
            plt.xlabel("Time")
            plt.ylabel("MWh")
            plt.title("Plot of true active_losses and predictions for split " + str(split_idx + 1) + ".")
            
            # Save the plot with a dynamic filename
            plt.savefig(f"plots/line_plot_{split_idx + 1}.png")

            # Close the plot to prevent overlapping when creating the next plot
            plt.close()