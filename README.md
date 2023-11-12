# Active_losses_Swissgrid

## Overview
Swissgrid took part to a Machine Learning Hackaton with the goal of predicting the active losses for the grid. The result were impressived and can be found here [here](https://github.com/Swissgrid-AG-External/energydatahackdays23/tree/d5f88f3ff117ffcaafd43167e6357f7a5bfbc4a2/group3). The results were very impressive and many stakeholders were keen to see the results on internal data instead of old data.

This repository corresponds to a proposal of solution for the ML Engineer Challenge that can be found [here](https://github.com/Swissgrid-AG-External/coding_challenges/blob/main/ml_engineer/intern/README.md). The 3 main goals of this challenge are the followings:
1. write a markdown file (Approach.md) to describe the approach 
2. Refactor the notebooks into clean, well-organized pyhton file while preserving existing functionnalities. The primary goal was to enhance code structure, readability, and maintainability.
3. Change or add 1 or 2 functions to the script


## Installation
Python 3.11.5 was used for this project
- conda env create -f environment.yaml
- conda activate active_loss

## Structure of the code

```
├── README.md
├── Approach.md
├── __init__.py
├── data
│   ├── processed
│   │   └── df_base_trainval_preprocessed.csv
│   └── raw
│       ├── Active-Losses-2019-2021.csv
│       ├── NTC_2019_2021.csv
│       ├── data_explanation.md
│       ├── eq_renewables_2019-2021.csv
│       └── eq_temp_2019-2021.csv
├── main.py
├── parameter_tuning.py
├── pipeline.py
├── preprocessing.py
├── project_structure.txt
├── training.py
├── util.py
├── environment.yml
├── saved_models
│   ├── EN_hyperparams.pickle
│   ├── en_trainval.pickle
│   ├── en_tuning_optuna.csv
│   └── sklearn_base_pipeline_unfitted.pickle
└── notebooks
   ├── 0_Data Preprocessing Base.ipynb
   └── 2_ElasticNet.ipynb
```


## Data
```

├── data
│   ├── processed
│   │   └── df_base_trainval_preprocessed.csv
│   └── raw
│       ├── Active-Losses-2019-2021.csv
│       ├── NTC_2019_2021.csv
│       ├── data_explanation.md
│       ├── eq_renewables_2019-2021.csv
│       └── eq_temp_2019-2021.csv
```


