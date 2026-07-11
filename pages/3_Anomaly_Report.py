import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")


@st.cache_data
def load_anomalies():
    return pd.read_csv("artifacts/anomalies.csv", parse_dates=["Month"])


df = load_anomalies()

st.title("🚨 Anomaly Report")
st.caption(
    "Monthly sales are decomposed into Trend + Seasonal + Residual components (STL "
    "decomposition) to account for strong seasonality. A month is flagged as an anomaly "
    "when its residual (the part of sales unexplained by trend/seasonality) is an outlier."
)

anomalies = df[df["Is_Anomaly"]]

col1, col2 = st.columns(2)
col1.metric("Total Months Analyzed", len(df))
col2.metric("Anomalies Detected", len(anomalies))

st.divider()

# ---- anomaly chart ----
st.subheader("Sales Over Time — Anomalies Highlighted")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["Month"],
        y=df["Sales"],
        mode="lines",
        name="Monthly Sales",
        line=dict(color="#2E86AB"),
    )
)
fig.add_trace(
    go.Scatter(
        x=df["Month"],
        y=df["Trend"],
        mode="lines",
        name="Trend",
        line=dict(color="#95A5A6", dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=anomalies["Month"],
        y=anomalies["Sales"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="#E74C3C", size=13, symbol="x"),
    )
)
fig.update_layout(
    xaxis_title="Month", yaxis_title="Sales ($)", hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- residual chart ----
with st.expander("Show residual (deseasonalized) chart used for detection"):
    fig_resid = go.Figure()
    fig_resid.add_trace(
        go.Bar(
            x=df["Month"],
            y=df["Residual"],
            marker_color=[
                "#E74C3C" if a else "#BDC3C7" for a in df["Is_Anomaly"]
            ],
            name="Residual",
        )
    )
    fig_resid.update_layout(xaxis_title="Month", yaxis_title="Residual ($)")
    st.plotly_chart(fig_resid, use_container_width=True)

# ---- anomaly table ----
st.subheader("Detected Anomaly Dates")
if len(anomalies):
    table = anomalies[["Month", "Sales", "Z_Score"]].copy()
    table["Month"] = table["Month"].dt.strftime("%B %Y")
    table["Sales"] = table["Sales"].map(lambda v: f"${v:,.2f}")
    table["Z_Score"] = table["Z_Score"].round(2)
    table = table.rename(columns={"Z_Score": "Residual Z-Score"})
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("No anomalies detected at the current threshold.")
