import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/"
    "COVID-19/master/csse_covid_19_data/"
    "csse_covid_19_time_series/"
    "time_series_covid19_deaths_US.csv"
)

CACHE_PATH = "data/raw/jhu_deaths_us.csv"


def load_raw_data(cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:

    if cache and not force_refresh and os.path.exists(CACHE_PATH):
        logging.info("Loading cached raw data.")
        try:
            return pd.read_csv(CACHE_PATH)
        except Exception as e:
            logging.warning("Cached file corrupted. Re-downloading.")
    
    logging.info("Downloading JHU dataset.")
    
    try:
        df = pd.read_csv(JHU_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

    if cache:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        logging.info("Raw data cached locally.")

    return df