import pandas as pd


def create_national_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date")["daily_deaths"].sum().reset_index()


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_mean_7"] = df["daily_deaths"].rolling(7).mean()
    df["rolling_std_7"] = df["daily_deaths"].rolling(7).std()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df