import os
import pandas as pd

JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/"
    "COVID-19/master/csse_covid_19_data/"
    "csse_covid_19_time_series/"
    "time_series_covid19_deaths_US.csv"
)

CACHE_PATH = "data/raw/jhu_deaths_us.csv"


def load_raw_data(cache: bool = True) -> pd.DataFrame:
    if cache and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)

    df = pd.read_csv(JHU_URL)

    if cache:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)

    return df