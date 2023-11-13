"""
This file contains the pipeline for the model.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from pickle import dump

def create_preprocessor(df: pd.DataFrame) -> Pipeline:
    """
    This function creates the preprocessor pipeline for the model.
    :param df: dataframe
    :return: pipeline
    """
    # Drop unnecessary columns
    drop_cols = ["Zeitstempel", "MWh"]
    df = df.drop(drop_cols, axis=1)

    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )

    # Making column transformer where all transformers in the pipelines are included
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, num_cols),
            ("categorical", categorical_transformer, cat_cols),
        ],
        remainder="passthrough",
    )

    dump(preprocessor, open('saved_models/sklearn_base_pipeline_unfitted.pickle', 'wb'))

    return preprocessor


if __name__ == "__main__":
    df = pd.read_csv("data/processed/df_preprocessed.csv")
    preprocessor = create_preprocessor(df)
    preprocessor.fit(df)