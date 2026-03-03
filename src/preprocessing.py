import os
import pandas as pd
import re

PROCESSED_PATH = "data/processed/preprocessed_jhu.csv"

def preprocess_jhu_data(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:

    if save and os.path.exists(PROCESSED_PATH):
        return pd.read_csv(PROCESSED_PATH, parse_dates=["date"])

    df = df[df["Country_Region"] == "US"].copy()

    date_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2}")
    date_cols = [col for col in df.columns if date_pattern.match(col)]

    df_state = (
        df.groupby("Province_State")[date_cols + ["Population"]]
        .sum()
        .reset_index()
    )

    df_long = df_state.melt(
        id_vars=["Province_State", "Population"],
        value_vars=date_cols,
        var_name="date",
        value_name="cumulative_deaths"
    )

    df_long["date"] = pd.to_datetime(df_long["date"], format="%m/%d/%y")
    df_long = df_long.sort_values(["Province_State", "date"])

    df_long["daily_deaths"] = (
        df_long.groupby("Province_State")["cumulative_deaths"]
        .diff()
        .fillna(0)
        .clip(lower=0)
    )

    if save:
        os.makedirs("data/processed", exist_ok=True)
        df_long.to_csv(PROCESSED_PATH, index=False)

    return df_long