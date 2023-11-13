"""
Main file for the project
"""

from src.util import load_processed_data, create_time_series_splits, analyze_optuna_study
from src.training import train_final
from src.preprocessing import preprocess_data
from src.parameter_tuning import objective
from src.pipeline import create_preprocessor
from src.visualization import visualize_splits, visualize_model_predictions
import optuna
import pandas as pd



if __name__ == '__main__':
  
    # prepocess the data and save it in csv file
    df = preprocess_data()
    # Alternatively, load alerady preprocessed data
    # df = pd.read_csv("data/processed/df_preprocessed.csv")

    # Load preprocessed data
    X, y = load_processed_data(df)

    # Create time series splits
    splits = create_time_series_splits(X)

    # Visualize splits by saving 5 plots in the plots folder
    visualize_splits(splits, df, y)

    # Fit preprocessor on the entire dataset
    preprocessor = create_preprocessor(df)

    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, preprocessor, splits), n_trials=2, timeout=3600*0.5)

    # Visualize the prediction of the model on the testing data with the best hyperparameters
    visualize_model_predictions(splits, X, y, preprocessor, study.best_params)

    # Analyze Optuna study by outputing the numbeer of trials, the minimal MAE and the best hyperparameters
    analyze_optuna_study(study)

    # Train the final model with the best hyperparameters on the entire dataset. And save it in saved_models for later use
    final_model = train_final(X, y, preprocessor, study.best_params)
    