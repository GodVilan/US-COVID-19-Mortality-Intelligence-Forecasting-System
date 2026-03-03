import pandas as pd
from src.preprocessing import preprocess_jhu_data


def test_preprocess_jhu_data():
    df = pd.DataFrame({
        "Province_State": ["Test"],
        "Population": [100000],
        "Country_Region": ["US"],
        "1/1/20": [0],
        "1/2/20": [1]
    })

    processed = preprocess_jhu_data(df)
    assert "daily_deaths" in processed.columns