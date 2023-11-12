"""
Main file for the project
"""

# main.py
from util import load_processed_data, create_time_series_splits, train_preprocessor
# from src.visualization import visualize_splits
from training import train_and_evaluate_model, train_final
from parameter_tuning import objective
import optuna
import warnings
from plotnine import ggplot, aes, labs, geom_line, scale_colour_manual, theme_light, theme
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from pickle import dump, load
import numpy as np


import sys
print(sys.path)
# Load data
X, y = load_processed_data('data/processed/df_base_trainval_preprocessed.csv')

# Create time series splits
splits = create_time_series_splits(X)

# Visualize splits
# visualize_splits(splits, df, y)

# Fit preprocessor on the entire dataset
preprocessor = load(open('saved_models/sklearn_base_pipeline_unfitted.pickle', 'rb'))

# Hyperparameter tuning using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y, preprocessor, splits), n_trials=2, timeout=3600*0.5)

# Get best hyperparameters
best_params = study.best_params
print(best_params)
best_alpha = best_params['alpha']
best_l1_ratio = best_params['l1_ratio']

# Train the final model with the best hyperparameters on the entire dataset
final_evaluation_score = train_final(X, y, preprocessor, best_params)
print(f'Final Model Evaluation Score: {final_evaluation_score}')

# Save results
pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

# Display the study statistics
print("\nStudy statistics: ")
print(f"  Number of finished trials: {len(study.trials)}")
print(f"  Number of pruned trials: {len(pruned_trials)}")
print(f"  Number of complete trials: {len(complete_trials)}")

study.trials_dataframe().to_csv("en_tuning_optuna.csv", index=False)

with open("EN_hyperparams.pickle", 'wb') as file:
    dump(study.best_params, file)


if __name__ == '__main__':
    print("seems ok")
    pass