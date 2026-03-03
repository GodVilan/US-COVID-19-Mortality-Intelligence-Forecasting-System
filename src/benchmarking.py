import numpy as np
import pandas as pd

from .modeling import train_sarimax, train_prophet, train_xgboost
from .evaluation import evaluate_model
from .config import TEST_SIZE


def benchmark_models(national_df: pd.DataFrame):

    results = {}

    train = national_df[:-TEST_SIZE]
    test = national_df[-TEST_SIZE:]

    # =========================
    # 1️⃣ SARIMAX
    # =========================
    train_log = np.log1p(train["daily_deaths"])
    sarimax_model = train_sarimax(train_log)

    # Get forecast object
    forecast_obj = sarimax_model.get_forecast(steps=TEST_SIZE)

    forecast_log = forecast_obj.predicted_mean
    conf_int_log = forecast_obj.conf_int()

    # Inverse transform
    sarimax_forecast = np.expm1(forecast_log)
    sarimax_forecast = np.maximum(sarimax_forecast, 0)

    lower_ci = np.expm1(conf_int_log.iloc[:, 0])
    upper_ci = np.expm1(conf_int_log.iloc[:, 1])

    lower_ci = np.maximum(lower_ci, 0)
    upper_ci = np.maximum(upper_ci, 0)

    results["SARIMAX"] = evaluate_model(
        test["daily_deaths"],
        sarimax_forecast
    )

    # =========================
    # 2️⃣ Prophet
    # =========================
    prophet_model = train_prophet(train.copy())

    future = prophet_model.make_future_dataframe(periods=TEST_SIZE)
    forecast_df = prophet_model.predict(future)

    prophet_forecast = forecast_df["yhat"].iloc[-TEST_SIZE:].values
    prophet_forecast = np.maximum(prophet_forecast, 0)

    results["Prophet"] = evaluate_model(
        test["daily_deaths"],
        prophet_forecast
    )

    # =========================
    # 3️⃣ XGBoost
    # =========================
    df_ml = national_df.copy()
    df_ml["lag_1"] = df_ml["daily_deaths"].shift(1)
    df_ml["lag_7"] = df_ml["daily_deaths"].shift(7)
    df_ml = df_ml.dropna()

    train_ml = df_ml[:-TEST_SIZE]
    test_ml = df_ml[-TEST_SIZE:]

    X_train = train_ml[["lag_1", "lag_7"]]
    y_train = train_ml["daily_deaths"]

    xgb_model = train_xgboost(X_train, y_train)

    # Recursive forecast
    history = train_ml.copy()
    xgb_predictions = []

    for i in range(TEST_SIZE):

        last_row = history.iloc[-1]
        lag_1 = last_row["daily_deaths"]
        lag_7 = history.iloc[-7]["daily_deaths"]

        X_input = np.array([[lag_1, lag_7]])
        pred = xgb_model.predict(X_input)[0]
        pred = max(pred, 0)

        xgb_predictions.append(pred)

        new_row = {
            "date": test.iloc[i]["date"],
            "daily_deaths": pred,
            "lag_1": lag_1,
            "lag_7": lag_7
        }

        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True
        )

    xgb_forecast = np.array(xgb_predictions)

    results["XGBoost"] = evaluate_model(
        test["daily_deaths"],
        xgb_forecast
    )

    # =========================
    # Final Selection
    # =========================
    results_df = pd.DataFrame(results).T.sort_values("MAE")

    # Select best model (lowest MAE)
    best_model_name = results_df.index[0]

    if best_model_name == "SARIMAX":
        final_forecast = sarimax_forecast
        final_model = sarimax_model
    elif best_model_name == "Prophet":
        final_forecast = prophet_forecast
        final_model = prophet_model
    else:
        final_forecast = xgb_forecast
        final_model = xgb_model

    residuals = test["daily_deaths"] - final_forecast

    return results_df, sarimax_forecast, residuals, sarimax_model, lower_ci, upper_ci

def rolling_cross_validation(national_df, window_size=30, folds=5):

    errors = []

    total_length = len(national_df)

    for i in range(folds):

        end_train = total_length - window_size * (folds - i)
        start_test = end_train
        end_test = start_test + window_size

        train = national_df[:end_train]
        test = national_df[start_test:end_test]

        train_log = np.log1p(train["daily_deaths"])
        model = train_sarimax(train_log)

        forecast_log = model.forecast(steps=window_size)
        forecast = np.expm1(forecast_log)
        forecast = np.maximum(forecast, 0)

        metrics = evaluate_model(test["daily_deaths"], forecast)
        errors.append(metrics["MAE"])

    return np.mean(errors)