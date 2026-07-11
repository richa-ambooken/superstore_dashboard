import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Forecast Explorer", page_icon="🔮", layout="wide")


@st.cache_data
def load_forecasts():
    with open("artifacts/forecasts.json") as f:
        forecasts = json.load(f)
    with open("artifacts/forecast_metrics.json") as f:
        metrics = json.load(f)
    return forecasts, metrics


forecasts, metrics = load_forecasts()

st.title("🔮 Forecast Explorer")
st.caption(
    "Forecasts generated with a Holt-Winters (triple exponential smoothing) model, "
    "trained per Category / Region on the full monthly sales history."
)

# ---- controls ----
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    dimension = st.selectbox("Select dimension", ["Category", "Region"])

available_values = sorted(
    {k.split("|")[1] for k in forecasts.keys() if k.startswith(f"{dimension}|")}
)

with col2:
    value = st.selectbox(f"Select {dimension}", available_values)

with col3:
    horizon = st.select_slider(
        "Forecast horizon (months ahead)",
        options=[1, 2, 3],
        value=3,
    )

key = f"{dimension}|{value}"
data = forecasts[key]
model_metrics = metrics[key]

hist_dates = pd.to_datetime(data["history_dates"])
hist_values = data["history_values"]
fcst_dates = pd.to_datetime(data["forecast_dates"])[:horizon]
fcst_values = data["forecast_values"][:horizon]

# ---- chart ----
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist_dates,
        y=hist_values,
        mode="lines",
        name="Historical Sales",
        line=dict(color="#2E86AB"),
    )
)
# connect last historical point to first forecast point for a continuous line
bridge_x = [hist_dates[-1]] + list(fcst_dates)
bridge_y = [hist_values[-1]] + list(fcst_values)
fig.add_trace(
    go.Scatter(
        x=bridge_x,
        y=bridge_y,
        mode="lines+markers",
        name=f"Forecast (+{horizon}mo)",
        line=dict(color="#E67E22", dash="dash"),
    )
)
fig.update_layout(
    title=f"{dimension}: {value} — Sales Forecast",
    xaxis_title="Month",
    yaxis_title="Sales ($)",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---- forecast table ----
st.subheader("Forecast Values")
fcst_table = pd.DataFrame(
    {
        "Month": [d.strftime("%b %Y") for d in fcst_dates],
        "Forecasted Sales ($)": [f"{v:,.2f}" for v in fcst_values],
    }
)
st.dataframe(fcst_table, use_container_width=True, hide_index=True)

# ---- model performance ----
st.subheader("Model Performance (Backtest on last 3 months of history)")
m1, m2 = st.columns(2)
mae = model_metrics["MAE"]
rmse = model_metrics["RMSE"]
m1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}" if mae is not None else "N/A")
m2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}" if rmse is not None else "N/A")

st.caption(
    "MAE/RMSE are computed by holding out the last 3 months of actual data, "
    "forecasting those months, and comparing the forecast to what actually happened. "
    "Lower values indicate a more accurate model."
)
