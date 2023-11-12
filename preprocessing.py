"""
This file contains the preprocessing functions for the data.
- loadig the data
- adding features
"""


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


def get_data(file_paths=None) -> pd.DataFrame:
    """
    This function gets the data from the data folder and returns a merged dataframe.
    :param file_paths: Dictionary of file paths.
    :return: df: dataframe
    """
    if file_paths is None:
        file_paths = {
            "active_losses": "data/raw/Active-Losses.csv",
            "renewables": "data/raw/eq_renewables.csv",
            "temperature": "data/raw/eq_temp.csv",
            "ntc": "data/raw/NTC.csv",
        }
    active_losses = pd.read_csv(file_paths["active_losses"])
    
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
        file_paths["renewables"], parse_dates=["datetime"]
    ).pipe(clean_columns)

    temperature = pd.read_csv(
        file_paths["temperature"], parse_dates=["datetime"]
    ).assign(datetime=lambda x: x["datetime"] - pd.Timedelta("1 hour"))

    ntc = pd.read_csv(file_paths["ntc"], parse_dates=["datetime"])

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

class FeatureEngineer:
    """
    This class contains the feature engineering functions.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sin_transformer(self, x, period):
        return np.sin(x / period * 2 * np.pi)

    def cos_transformer(self, x, period):
        return np.cos(x / period * 2 * np.pi)

    def add_date_features(self):
        self.df = self.df.assign(
            day=lambda x: x.Zeitstempel.dt.date,
            date=lambda x: x.Zeitstempel.dt.date.astype(str)
        )

    def add_time_features(self):
        self.df = self.df.assign(
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

    def add_holiday_features(self):
        country = "CH"
        regional_holidays = holidays.CH(
            years=self.df.Zeitstempel.dt.year.unique().tolist()
        )

        holiday_df = pd.DataFrame(
            {
                "holiday_name": list(regional_holidays.values()),
                "holiday_date": list(regional_holidays.keys()),
            }
        )

        self.df = (
            self.df.merge(holiday_df, how="left", left_on="day", right_on="holiday_date")
            .drop(["holiday_date", "day"], axis=1)
            .assign(holiday_name=lambda x: np.where(x["holiday_name"].isna(), "none", x["holiday_name"]))
        )

    def add_cyclical_features(self):
        self.df["hour_sin"] = self.sin_transformer(self.df["hour"].astype(float), 24)
        self.df["hour_cos"] = self.cos_transformer(self.df["hour"].astype(float), 24)

        self.df["week_hour_sin"] = self.sin_transformer(self.df["week_hour"].astype(float), 168)
        self.df["week_hour_cos"] = self.cos_transformer(self.df["week_hour"].astype(float), 168)

        self.df["month_sin"] = self.sin_transformer(self.df["month"].astype(float), 12)
        self.df["month_cos"] = self.cos_transformer(self.df["month"].astype(float), 12)

        self.df["quarter_sin"] = self.sin_transformer(self.df["quarter"].astype(float), 4)
        self.df["quarter_cos"] = self.cos_transformer(self.df["quarter"].astype(float), 4)

        self.df["wday_sin"] = self.sin_transformer(self.df["wday"].astype(float), 7)
        self.df["wday_cos"] = self.cos_transformer(self.df["wday"].astype(float), 7)

        self.df = self.df.drop(["hour", "month", "quarter", "wday", "week_hour"], axis=1)

    def add_lag_features(self):
        days = self.df["Zeitstempel"].tolist()
        lag_lists = []

        for day_idx, day in enumerate(tqdm(days)):
            prior_cutoff = (day - pd.Timedelta("1 day")).strftime('%Y-%m-%d')
            lags = self.df.query("date <= @prior_cutoff").tail(168)["MWh"].tolist()
            lag_lists.append(lags)

        lag_df = pd.DataFrame(lag_lists).add_prefix("target_lag_")
        self.df = pd.concat([self.df.drop("date", axis=1), lag_df], axis=1)
        columns_to_move = ["Zeitstempel", "MWh"]
        self.df = self.df[columns_to_move + [col for col in self.df.columns if col not in columns_to_move]]

    def execute_feature_engineering(self):
        self.add_date_features()
        self.add_time_features()
        self.add_holiday_features()
        self.add_cyclical_features()
        self.add_lag_features()

        return self.df


def preprocess_data(file_paths=None):
    # Get the data
    df = get_data(file_paths)

    # Feature engineering
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.execute_feature_engineering()

    # Save the resulting dataframe
    df.to_csv("data/processed/df_preprocessed.csv", index=False)

    return df

if __name__ == "__main__":
    preprocess_data()