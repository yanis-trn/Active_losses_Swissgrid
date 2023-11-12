"""
Main file for the project
"""

from util import load_processed_data, create_time_series_splits, train_preprocessor, analyze_optuna_study
from training import train_and_evaluate_model, train_final
from preprocessing import preprocess_data
from parameter_tuning import objective
from pipeline import create_preprocessor
import optuna
import warnings
from plotnine import ggplot, aes, labs, geom_line, scale_colour_manual, theme_light, theme
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from pickle import dump, load
import numpy as np
import pandas as pd



if __name__ == '__main__':
  
    # prepocess the data and save it in csv file
    # df = preprocess_data()
    # Alternatively, load alerady preprocessed data
    df = pd.read_csv("data/processed/df_preprocessed.csv")

    # Load preprocessed data
    X, y = load_processed_data(df)

    # Create time series splits
    splits = create_time_series_splits(X)

    # Visualize splits
    # visualize_splits(splits, df, y)

    # Fit preprocessor on the entire dataset
    preprocessor = create_preprocessor(df)

    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, preprocessor, splits), n_trials=2, timeout=3600*0.5)
    # print(f'Final Model Evaluation Score: {final_evaluation_score}')

    analyze_optuna_study(study)

    # # Save results
    # pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    # best_trial = study.best_trial

    # # Access the minimal MAE obtained during the training and the best hyperparameters
    # min_mae = best_trial.value
    # best_params = study.best_params


    # # Display the study statistics
    # print("\nStudy statistics: ")
    # print(f"  Number of finished trials: {len(study.trials)}")
    # print(f"  Number of pruned trials: {len(pruned_trials)}")
    # print(f"  Number of complete trials: {len(complete_trials)}")

    # print(f"Minimal MAE: {min_mae}")
    # print(best_params)

    # study.trials_dataframe().to_csv("saved_models/trials_tuning_optuna.csv", index=False)

    # with open("saved_models/best_model_hyperparams.pickle", 'wb') as file:
    #     dump(study.best_params, file)



    # Train the final model with the best hyperparameters on the entire dataset

    final_model = train_final(X, y, preprocessor, study.best_params)

    # Save the final model for later use
    with open("saved_models/final_model.pickle", 'wb') as file:
        dump(final_model, file)