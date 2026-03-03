from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb


def train_sarimax(series):
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def train_prophet(df):
    prophet_df = df.rename(columns={"date": "ds", "daily_deaths": "y"})
    model = Prophet()
    model.fit(prophet_df)
    return model


def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model