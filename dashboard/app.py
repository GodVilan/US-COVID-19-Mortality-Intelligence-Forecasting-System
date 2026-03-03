import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dash
from dash import html, dcc
import dash_table

from src.ingestion import load_raw_data
from src.preprocessing import preprocess_jhu_data
from src.feature_engineering import create_national_aggregation
from src.benchmarking import benchmark_models, rolling_cross_validation
from src.visualization import plot_trend, plot_forecast, plot_residuals
from src.config import TEST_SIZE


# ===============================
# KPI CARD
# ===============================
def kpi_card(title, value):
    return html.Div(
        style={
            "flex": "1",
            "backgroundColor": "white",
            "padding": "18px",
            "borderRadius": "10px",
            "boxShadow": "0 3px 8px rgba(0,0,0,0.05)",
            "textAlign": "center"
        },
        children=[
            html.H4(
                title,
                style={"marginBottom": "8px", "fontWeight": "500"}
            ),
            html.H2(
                value,
                style={
                    "color": "#2563eb",
                    "fontWeight": "600",
                    "margin": "0"
                }
            )
        ]
    )


# ===============================
# Data Pipeline
# ===============================
df = load_raw_data()
df = preprocess_jhu_data(df)
national = create_national_aggregation(df)

results_df, forecast, residuals, _, lower_ci, upper_ci = benchmark_models(national)
results_df = results_df.round(2)
cv_mae = rolling_cross_validation(national)

trend_fig = plot_trend(national)
forecast_fig = plot_forecast(national, forecast, lower_ci, upper_ci, TEST_SIZE)
residual_fig = plot_residuals(
    national["date"].iloc[-TEST_SIZE:], residuals
)

best_model = results_df.index[0]
best_mae = results_df.iloc[0]["MAE"]


app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "padding": "36px",
        "backgroundColor": "#f3f4f6",  # 🔥 Softer SaaS gray
        "fontFamily": "Inter, Segoe UI, Arial"
    },
    children=[

        html.H1(
            "US COVID-19 Mortality Intelligence & Forecasting System",
            style={
                "textAlign": "center",
                "marginBottom": "28px",
                "fontWeight": "600"  # 🔥 Stronger title
            }
        ),

        # KPI SECTION
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "28px"},
            children=[
                kpi_card("Best Model", best_model),
                kpi_card("Holdout MAE", f"{best_mae:.2f}"),
                kpi_card("Rolling CV MAE", f"{cv_mae:.2f}"),
                kpi_card("Test Window", f"{TEST_SIZE} Days"),
            ]
        ),

        html.Hr(style={"margin": "28px 0"}),

        html.H3("Model Performance Comparison",
                style={"marginBottom": "12px"}),

        dash_table.DataTable(
            data=results_df.reset_index().to_dict("records"),
            columns=[
                {"name": "Model", "id": "index"},
                {"name": "MAE", "id": "MAE"},
                {"name": "RMSE", "id": "RMSE"},
                {"name": "SMAPE (%)", "id": "SMAPE (%)"}
            ],
            style_table={"overflowX": "auto", "marginBottom": "24px"},
            style_cell={
                "textAlign": "center",
                "padding": "10px"
            },
            style_header={
                "backgroundColor": "#111827",
                "color": "white",
                "fontWeight": "600"
            },
            style_data_conditional=[
                {
                    "if": {"row_index": 0},
                    "backgroundColor": "#dcfce7",
                    "fontWeight": "600"
                }
            ]
        ),

        html.Hr(style={"margin": "28px 0"}),

        dcc.Graph(figure=trend_fig),

        html.Hr(style={"margin": "28px 0"}),

        dcc.Graph(figure=forecast_fig),

        html.Hr(style={"margin": "28px 0"}),

        dcc.Graph(figure=residual_fig),
    ]
)


# if __name__ == "__main__":
#     app.run(debug=True)
#     import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)