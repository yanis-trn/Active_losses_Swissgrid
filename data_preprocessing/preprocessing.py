"""
This file contains the preprocessing functions for the data.
- loadig the data
- adding features
"""

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import config_context

import pandas as pd
import numpy as np
from plotnine import *
from mizani.formatters import comma_format, percent_format, currency_format
from datetime import datetime, timedelta, date
from tqdm.notebook import tqdm
from skimpy import clean_columns
from IPython.display import clear_output, display
import holidays

from pickle import dump, load


def get_data() -> pd.DataFrame:
    """
    This function gets the data from the data folder and returns a merged dataframe.
    :return: df: dataframe
    """
    active_losses = pd.read_csv("data/raw/Active-Losses-2019-2021.csv")
    
    active_losses.columns = active_losses.iloc[
    0,
    ]
    active_losses = (
        active_losses.drop(active_losses.index[0])
        .assign(
            Zeitstempel=lambda x: (
                pd.to_datetime(x["Zeitstempel"]) - pd.Timedelta("15 minutes")
            ).dt.floor(freq="H"),
            kWh=lambda x: pd.to_numeric(x["kWh"]) / 1000,
        )
        .groupby("Zeitstempel")
        .agg(MWh=("kWh", "sum"))
        .reset_index()
    )
    renewables = pd.read_csv(
        "data/raw/eq_renewables_2019-2021.csv", parse_dates=["datetime"]
    ).pipe(clean_columns)

    temperature = pd.read_csv(
        "data/raw/eq_temp_2019-2021.csv", parse_dates=["datetime"]
    ).assign(datetime=lambda x: x["datetime"] - pd.Timedelta("1 hour"))

    ntc = pd.read_csv("data/raw/NTC_2019_2021.csv", parse_dates=["datetime"])

    df = (
        active_losses
        .merge(
            temperature, how="left", left_on="Zeitstempel", right_on="datetime"
        )
        .drop("datetime", axis=1)
        .ffill()
        .merge(renewables, how="left", left_on="Zeitstempel", right_on="datetime")
        .drop("datetime", axis=1)
        .merge(ntc, how="left", left_on="Zeitstempel", right_on="datetime")
        .drop("datetime", axis=1)
    )

    return df

def add_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function adds features to the dataframe.
    :param df: dataframe
    :return: df: dataframe
    """
    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    country = "CH"
    regional_holidays = holidays.CH(
        years=df.Zeitstempel.dt.year.unique().tolist()
    )

    holiday_df = pd.DataFrame(
        {
            "holiday_name": list(regional_holidays.values()),
            "holiday_date": list(regional_holidays.keys()),
        }
    )
    
    df = (
        df.assign(
            hour=lambda x: x.Zeitstempel.dt.hour + 1,
            month=lambda x: x.Zeitstempel.dt.month,
            quarter=lambda x: x.Zeitstempel.dt.quarter,
            wday=lambda x: x.Zeitstempel.dt.day_of_week + 1,
            weekend=lambda x: np.where(
                x.Zeitstempel.dt.day_name().isin(["Sunday", "Saturday"]), 1, 0
            ).astype(str),
            work_hour=lambda x: np.where(
                x["hour"].isin([19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7]), 0, 1
            ).astype(str),
            week_hour=lambda x: x.Zeitstempel.dt.dayofweek * 24 + (x.Zeitstempel.dt.hour + 1),
            year=lambda x: x.Zeitstempel.dt.year,
            hour_counter=lambda x: np.arange(0, x.shape[0])
        )
        .assign(day=lambda x: x.Zeitstempel.dt.date)
        .merge(holiday_df, how="left", left_on="day", right_on="holiday_date")
        .drop(["holiday_date", "day"], axis=1)
        .assign(holiday_name = lambda x: np.where(x["holiday_name"].isna(), "none", x["holiday_name"]))
    )
    # hour in day
    df["hour_sin"] = sin_transformer(24).fit_transform(df["hour"].astype(float))
    df["hour_cos"] = cos_transformer(24).fit_transform(df["hour"].astype(float))

    # hour in week
    df["week_hour_sin"] = sin_transformer(168).fit_transform(df["week_hour"].astype(float))
    df["week_hour_cos"] = cos_transformer(168).fit_transform(df["week_hour"].astype(float))

    # month
    df["month_sin"] = sin_transformer(12).fit_transform(df["month"].astype(float))
    df["month_cos"] = cos_transformer(12).fit_transform(df["month"].astype(float))

    # quarter
    df["quarter_sin"] = sin_transformer(4).fit_transform(df["quarter"].astype(float))
    df["quarter_cos"] = cos_transformer(4).fit_transform(df["quarter"].astype(float))

    # weekday
    df["wday_sin"] = sin_transformer(7).fit_transform(df["wday"].astype(float))
    df["wday_cos"] = cos_transformer(7).fit_transform(df["wday"].astype(float))

    df = df.drop(["hour", "month", "quarter", "wday", "week_hour"], axis=1)

    days = df["Zeitstempel"].tolist()

    df = df.assign(date = lambda x: x.Zeitstempel.dt.date.astype(str))

    prior_cutoff = (days[23] - pd.Timedelta("1 day")).strftime('%Y-%m-%d')
    f"{prior_cutoff} 23:00"

    lag_lists = []

    for day_idx, day in enumerate(tqdm(days)):
        prior_cutoff = (day - pd.Timedelta("1 day")).strftime('%Y-%m-%d')

        lags = df.query("date <= @prior_cutoff").tail(168)["MWh"].tolist()
        lag_lists.append(lags)

    lag_df = pd.DataFrame(lag_lists).add_prefix("target_lag_")
    
    df = pd.concat([df.drop("date", axis=1), lag_df], axis=1)

    return df


if __name__ == "__main__":
    # Test the get_data function
    df = get_data()

    df = add_features(df)
    # Display the first few rows of the resulting dataframe

    print(df.head())
