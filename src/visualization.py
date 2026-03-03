import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ===============================
# Historical Trend
# ===============================
def plot_trend(df):

    df = df.copy()
    df["rolling_mean_7"] = df["daily_deaths"].rolling(7).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["daily_deaths"],
        mode="lines",
        name="Daily Deaths",
        line=dict(color="#2563eb", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["rolling_mean_7"],
        mode="lines",
        name="7-Day Rolling Average",
        line=dict(color="#dc2626", width=3)
    ))

    fig.update_layout(
        template="plotly_white",
        height=480,
        title="Historical Mortality Trend",
        xaxis_title="Date",
        yaxis_title="Daily Deaths",
        font=dict(family="Inter, Segoe UI, Arial", size=14)
    )

    return fig


# ===============================
# Forecast (Prominent Section)
# ===============================
def plot_forecast(df, forecast, lower_ci, upper_ci, test_size):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["daily_deaths"],
        mode="lines",
        name="Actual",
        line=dict(color="#2563eb")
    ))

    fig.add_trace(go.Scatter(
        x=df["date"].iloc[-test_size:],
        y=forecast,
        mode="lines",
        name="Forecast",
        line=dict(color="#f59e0b", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df["date"].iloc[-test_size:],
        y=upper_ci,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df["date"].iloc[-test_size:],
        y=lower_ci,
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.2)",
        line=dict(width=0),
        name="95% Confidence Interval"
    ))

    max_val = df["daily_deaths"].max() * 1.2
    fig.update_yaxes(range=[0, max_val])

    fig.update_layout(
        template="plotly_white",
        height=520,  # 🔥 Taller forecast section
        title="30-Day Forecast with Uncertainty",
        xaxis_title="Date",
        yaxis_title="Daily Deaths",
        font=dict(family="Inter, Segoe UI, Arial", size=14)
    )

    return fig


# ===============================
# Residual Diagnostics
# ===============================
def plot_residuals(dates, residuals):

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Forecast Errors Over Time", "Residual Distribution")
    )

    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode="lines",
        line=dict(color="#7c3aed")
    ), row=1, col=1)

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        row=1,
        col=1
    )

    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=25,  # 🔥 Narrower bins
        marker_color="#7c3aed",
        opacity=0.75
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        font=dict(family="Inter, Segoe UI, Arial", size=14)
    )

    return fig