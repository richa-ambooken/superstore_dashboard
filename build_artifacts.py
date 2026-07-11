"""
build_artifacts.py
-------------------
Offline pipeline (run ONCE, before deploying) that:
  1. Loads and cleans the Superstore sales data (data/train.csv)
  2. Builds monthly aggregations (overall / by Category / by Region / by Sub-Category)
  3. Trains a Holt-Winters (Exponential Smoothing) forecast model for every
     Category and every Region, backtests it, and stores MAE / RMSE
  4. Runs anomaly detection (rolling z-score) on the overall monthly series
  5. Runs KMeans clustering on Sub-Category demand behaviour
  6. Saves every artifact into artifacts/ so the Streamlit app only has to
     read small pre-computed files at runtime (fast + Streamlit-Cloud-friendly)

Run with:  python build_artifacts.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

DATA_PATH = "data/train.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Order Date"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------------------
# 2. AGGREGATIONS FOR PAGE 1
# ---------------------------------------------------------------------------
def build_overview_aggregates(df):
    yearly = df.groupby("Year", as_index=False)["Sales"].sum().rename(
        columns={"Sales": "Total_Sales"}
    )
    yearly.to_csv(f"{ARTIFACT_DIR}/yearly_sales.csv", index=False)

    monthly = df.groupby("Month", as_index=False)["Sales"].sum().rename(
        columns={"Sales": "Total_Sales"}
    )
    monthly.to_csv(f"{ARTIFACT_DIR}/monthly_sales.csv", index=False)

    region_cat = df.groupby(["Region", "Category", "Sub-Category"], as_index=False)[
        "Sales"
    ].sum()
    region_cat.to_csv(f"{ARTIFACT_DIR}/region_category_sales.csv", index=False)

    # full monthly detail broken out by Region and by Category (used for filters)
    monthly_region = df.groupby(["Month", "Region"], as_index=False)["Sales"].sum()
    monthly_region.to_csv(f"{ARTIFACT_DIR}/monthly_sales_by_region.csv", index=False)

    monthly_category = df.groupby(["Month", "Category"], as_index=False)["Sales"].sum()
    monthly_category.to_csv(f"{ARTIFACT_DIR}/monthly_sales_by_category.csv", index=False)

    print("[1/4] Overview aggregates saved.")


# ---------------------------------------------------------------------------
# 3. FORECASTING (Category & Region), Holt-Winters, backtested MAE/RMSE
# ---------------------------------------------------------------------------
def make_monthly_series(df, dim, value):
    """Return a complete (no gaps) monthly series for a given Category/Region value."""
    sub = df[df[dim] == value]
    s = sub.groupby("Month")["Sales"].sum()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full_idx, fill_value=0.0)
    s.index.name = "Month"
    return s


def fit_and_forecast(series, horizon=3, test_size=3):
    """
    Fit Holt-Winters on the series minus the last `test_size` months (holdout),
    backtest on the holdout to get MAE/RMSE, then refit on the FULL series and
    forecast `horizon` months into the future.
    """
    series = series.astype(float)
    seasonal_periods = 12 if len(series) >= 24 else None
    seasonal = "add" if seasonal_periods else None

    # ---- backtest ----
    if len(series) > test_size + 6:
        train, test = series.iloc[:-test_size], series.iloc[-test_size:]
        try:
            model_bt = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            ).fit(optimized=True)
            preds = model_bt.forecast(test_size)
            mae = mean_absolute_error(test, preds)
            rmse = mean_squared_error(test, preds) ** 0.5
        except Exception:
            # fallback: naive last-value model
            preds = pd.Series([train.iloc[-1]] * test_size, index=test.index)
            mae = mean_absolute_error(test, preds)
            rmse = mean_squared_error(test, preds) ** 0.5
    else:
        mae, rmse = np.nan, np.nan

    # ---- refit on full data & forecast forward ----
    try:
        model_full = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        ).fit(optimized=True)
        future = model_full.forecast(horizon)
    except Exception:
        future = pd.Series([series.iloc[-1]] * horizon)

    future_idx = pd.date_range(
        series.index.max() + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    future.index = future_idx

    return future, mae, rmse


def build_forecasts(df):
    results = {}
    metrics = {}

    for dim in ["Category", "Region"]:
        for value in sorted(df[dim].unique()):
            series = make_monthly_series(df, dim, value)
            future, mae, rmse = fit_and_forecast(series, horizon=3, test_size=3)

            key = f"{dim}|{value}"
            results[key] = {
                "history_dates": [d.strftime("%Y-%m-%d") for d in series.index],
                "history_values": series.values.tolist(),
                "forecast_dates": [d.strftime("%Y-%m-%d") for d in future.index],
                "forecast_values": [round(float(v), 2) for v in future.values],
            }
            metrics[key] = {
                "MAE": None if np.isnan(mae) else round(float(mae), 2),
                "RMSE": None if np.isnan(rmse) else round(float(rmse), 2),
            }

    with open(f"{ARTIFACT_DIR}/forecasts.json", "w") as f:
        json.dump(results, f)
    with open(f"{ARTIFACT_DIR}/forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[2/4] Forecast models trained & saved for all Category/Region values.")


# ---------------------------------------------------------------------------
# 4. ANOMALY DETECTION
#    Monthly retail sales are strongly seasonal (Sep/Nov/Dec spikes every
#    year), so a plain rolling z-score gets "washed out" by that seasonality
#    and never flags anything. Instead we STL-decompose the series into
#    trend + seasonal + residual, then flag months whose RESIDUAL is an
#    outlier (|z-score of residual| > threshold). This finds months that are
#    unusual *after* accounting for the expected seasonal pattern.
# ---------------------------------------------------------------------------
def build_anomalies(df, z_thresh=1.5):
    from statsmodels.tsa.seasonal import STL

    monthly = df.groupby("Month")["Sales"].sum().sort_index()
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx, fill_value=0.0)

    stl = STL(monthly, period=12, robust=True).fit()
    resid = stl.resid
    trend = stl.trend
    seasonal = stl.seasonal

    resid_mean, resid_std = resid.mean(), resid.std()
    z_score = (resid - resid_mean) / resid_std
    is_anomaly = z_score.abs() > z_thresh

    anomaly_df = pd.DataFrame(
        {
            "Month": monthly.index,
            "Sales": monthly.values,
            "Trend": trend.values,
            "Seasonal": seasonal.values,
            "Residual": resid.values,
            "Z_Score": z_score.values,
            "Is_Anomaly": is_anomaly.values,
        }
    )
    anomaly_df.to_csv(f"{ARTIFACT_DIR}/anomalies.csv", index=False)
    print(
        f"[3/4] Anomaly detection complete. "
        f"{int(is_anomaly.sum())} anomalies flagged out of {len(monthly)} months."
    )


# ---------------------------------------------------------------------------
# 5. CLUSTERING (Product Demand Segments, on Sub-Category behaviour)
# ---------------------------------------------------------------------------
def build_clusters(df, n_clusters=3):
    monthly_sc = df.groupby(["Sub-Category", "Month"])["Sales"].sum().reset_index()

    features = []
    for sc, grp in monthly_sc.groupby("Sub-Category"):
        s = grp.set_index("Month")["Sales"].sort_index()
        total = s.sum()
        avg = s.mean()
        std = s.std(ddof=0) if len(s) > 1 else 0.0
        cv = std / avg if avg > 0 else 0.0  # coefficient of variation (volatility)
        # simple growth: compare first half vs second half average
        half = len(s) // 2 if len(s) > 1 else 1
        growth = (
            (s.iloc[half:].mean() - s.iloc[:half].mean()) / (s.iloc[:half].mean() + 1e-6)
            if len(s) > 1
            else 0.0
        )
        features.append(
            {
                "Sub-Category": sc,
                "Total_Sales": total,
                "Avg_Monthly_Sales": avg,
                "Volatility": cv,
                "Growth": growth,
            }
        )

    feat_df = pd.DataFrame(features)
    X = feat_df[["Total_Sales", "Avg_Monthly_Sales", "Volatility", "Growth"]].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df["Cluster"] = km.fit_predict(X_scaled)

    # Label clusters by average demand level (High / Medium / Low) based on Total_Sales
    order = (
        feat_df.groupby("Cluster")["Total_Sales"].mean().sort_values(ascending=False).index
    )
    label_map = {}
    labels = ["High Demand", "Medium Demand", "Low Demand"]
    for i, cluster_id in enumerate(order):
        label_map[cluster_id] = labels[i] if i < len(labels) else f"Segment {i+1}"
    feat_df["Segment"] = feat_df["Cluster"].map(label_map)

    feat_df.to_csv(f"{ARTIFACT_DIR}/clusters.csv", index=False)

    # also save PCA-free 2D projection (Total_Sales vs Avg_Monthly_Sales) for plotting
    print("[4/4] Clustering complete:")
    print(feat_df[["Sub-Category", "Segment"]].sort_values("Segment").to_string(index=False))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows, date range {df['Order Date'].min().date()} to "
          f"{df['Order Date'].max().date()}")

    build_overview_aggregates(df)
    build_forecasts(df)
    build_anomalies(df)
    build_clusters(df)

    print("\nAll artifacts saved to ./artifacts/")
