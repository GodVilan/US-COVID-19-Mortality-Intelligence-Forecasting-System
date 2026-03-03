# US COVID-19 Mortality Intelligence & Forecasting System

## Executive Summary
This project presents a production-grade end-to-end mortality forecasting system built using authoritative epidemiological data from the Johns Hopkins University COVID-19 repository. The objective is to benchmark statistical and machine learning models for short-term U.S. mortality forecasting and determine the most stable modeling strategy under seasonal epidemiological conditions.

The system evaluates Seasonal SARIMAX, Prophet, and XGBoost using holdout validation and rolling cross-validation. Seasonal SARIMAX achieved the strongest and most stable performance.

* **Holdout MAE:** 138.14
* **Rolling Cross-Validation MAE:** 137.86

The close alignment between holdout and cross-validation error confirms model robustness and low overfitting. This system demonstrates full-stack data science capability including ingestion, preprocessing, feature engineering, modeling, benchmarking, diagnostics, visualization, and deployment.

---

## Business Objective
Accurate short-term mortality forecasting supports:
* Public health resource allocation
* Hospital capacity planning
* Risk monitoring during outbreak volatility
* Epidemiological trend assessment

The goal was not simply to forecast, but to evaluate multiple modeling approaches and justify model selection based on quantitative evidence.

---

## Key Insights
* Mortality data exhibits strong weekly seasonality.
* Log transformation stabilizes variance and improves modeling reliability.
* Classical seasonal time-series modeling outperformed machine learning approaches in this structured epidemiological setting.
* Rolling cross-validation confirms performance stability.
* Residual diagnostics show no systematic bias and reasonable error distribution.

---

## Model Benchmarking Strategy
Three forecasting approaches were evaluated:
1. **Seasonal SARIMAX** (log-transformed)
2. **Prophet**
3. **XGBoost** with engineered lag features

### Evaluation Methodology
* 30-day holdout validation window
* Rolling cross-validation (5 folds)
* Metrics: MAE, RMSE, SMAPE

### Final Model Selection
SARIMAX was selected due to:
* Lowest holdout MAE and SMAPE.
* Stability under rolling cross-validation.
* Superior handling of weekly seasonality.
* Greater robustness under low-volume volatility.

---

## System Architecture

covid-mortality-intelligence/
├── data/
│   ├── raw/ (Not tracked)
│   ├── processed/ (Not tracked)
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── benchmarking.py
│   ├── visualization.py
│   ├── config.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── tests/
│   └── test_preprocessing.py
├── main.py
├── Dockerfile
├── render.yaml
└── requirements.txt

---

## Technical Components

### Data Ingestion
* Live pull from Johns Hopkins time-series dataset.
* Optional local caching and structured data loading.

### Preprocessing
* Wide-to-long transformation and state-level aggregation.
* Daily death computation from cumulative counts.
* Negative revision clipping and explicit date parsing.
* Sorting for temporal integrity.

### Feature Engineering
* National aggregation and rolling statistics (mean/std).
* Temporal encodings (day-of-week, month).
* Lag-based features for machine learning models.

### Modeling
* **Seasonal SARIMAX:** (1,1,1)(1,1,1,7)
* Log transformation with inverse exponential recovery.
* Non-negative forecast enforcement.
* Recursive forecasting for XGBoost.
* Prophet for additive trend modeling.

### Evaluation & Statistical Rigor
* Comprehensive metrics: MAE, RMSE, SMAPE.
* Rolling cross-validation (5 folds).
* Confidence intervals from SARIMAX forecast object.
* Residual time-series diagnostics and distribution visualization.

---

## Dashboard Capabilities
The interactive dashboard includes:
* Executive KPI summary and model comparison table.
* Historical mortality trend with rolling averages.
* 30-day forecast with 95% confidence intervals.
* Residual diagnostics (time-series + distribution).
* **UI/UX:** Clean SaaS-style layout with a structured visual hierarchy.

---

## Deployment
Dockerized for production portability with environment-aware port binding.

### Run Locally
Command 1: pip install -r requirements.txt
Command 2: python dashboard/app.py

### Run CLI Benchmarking
Command: python main.py

---

## Technologies Used
* **Core:** Python, Pandas, NumPy, Scikit-learn
* **Modeling:** Statsmodels, Prophet, XGBoost
* **Visualization:** Plotly, Dash
* **DevOps:** Docker

---

## Engineering Practices Demonstrated
* Modular architecture & separation of concerns.
* Config-driven design.
* Statistical validation and cross-validation rigor.
* Production deployment readiness & reproducibility.
* Unit testing.