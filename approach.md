# Approach to work on the ML Engineer Challenge - Swissgrid

## Main steps for solving the task:

1. Understand the nature of the data and the objective
2. Understand the approach of the group 3 to solve the tas
3. Note all the functionalities implemented 
4. Define the structure of the code
5. Implement the various functionalities in the various submodules and test them against original notebook
6. Write the main file to allow to run the entire process 
7. Rewrite the code for feature engineering into a clearer class 
8. Add a function to visualize forecasting


## Structural aspects:

### 1.Setting up a virtual environment 
Allows to manage project dependencies and packages properly. An environment.yaml file is created for easily exporting the active_loss virtual environment.

### 2.Setting up version control
Setting up of a git repository with best parctices, including commits with relevant messages.

### 3.Code structure
**file structure:** establishing a clear and intuitive file structure with related files grouped together.
**modularization:** Break down the previous notebook code into modular compenents. The goal is to improve code organization, readibility and reusability

### 4.Code documentation
**comments in the code:** ensure that the main different steps in the code are commented and write small description for each function, class for easier understanding 
**README.py** it should: describe the project, explain how to set up the virtual env and showcase the structure of the code / data

### 5.Implement logging
Ensure to save important metrics, models, hyperparameters and other relevant details. 
Output informational messages during the execution of the program for better undesrtanding and debugging

## ML frameworks:
Here are the main steps of this ML project and the used librairies:
- preprocessing the data and adding useful features using numpy, panda and skimpy
- creating a preprocessing pipeline sklearn
- creating time series split using sklearn
- hyperparameter tuning using optuna and ElasticNet model
- training ElasticNet on training data with the best hyperparameters and visualizing output on test data
- training the final model with the best hyperparameters on the entire dataset for later use

## Features added:

### FeatureEngineer() class:
Writting the FeatureEngineer class allowed better encapsulation of all functions (methods) for the creation of the different new features (time, dates, lags...). Defining all the methods in the same class gives a better understanding of all the features added and enables more modularrity in the future.

### train_and_visualize_model_predictions() function:
Different stakeholders being higly interested in the predictions from this model it appears that a simple function to visualize the accuracy of the forecast from the model could be useful. Once the splits are obtained from the preprocessed data and the preprocessor trained, the fucntion does the following:
- train a ElasticNet model using the previously best found hyperparameters
- get the prediction from the testing data
- plot the active loss predictions and true historical value 
- save those plots to /plots folder
Here is an example of obtained plot:
![Example plot](plots/line_plot_3.png)

## Potential additional work:
Due to time constraints, some good practices could not be implemented. Here is a non-exhaustive list of the the future work that could be done to ensure a more maintanable and adaptable code:
1. **Implement testing:** unit testing for functions and modules, integration testing between different componenents of the code
2. **Implement command-line argument parsing:** allowing for better adaptivity for the code. Making it possible to run the code easily with different data, model, hyperparameters...
3. **Implement configurations:** allows to quickly run the code with different configs 

## Feedback:

### Postive:
- very interesting subject with a real-world application
- open objectives that allow participants to come up with their own implementation of the solution
- easy to run notebooks and manageable data

### Less positive:
- staying within the 6h limit was tough, impacting the depth of the solution's completeness
