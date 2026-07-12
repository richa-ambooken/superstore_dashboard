"""
Superstore Sales Intelligence Dashboard
=========================================
A 4-page Streamlit app covering:
  1. Sales Overview Dashboard
  2. Forecast Explorer (Holt-Winters Exponential Smoothing)
  3. Anomaly Report (rolling z-score anomaly detection)
  4. Product Demand Segments (KMeans clustering of sub-categories)

Data source: Superstore-style order data (data/train.csv)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Superstore Sales Intelligence",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = "data/train.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data
def monthly_series(df: pd.DataFrame, group_col: str | None = None, group_val: str | None = None) -> pd.Series:
    """Aggregate Sales to a monthly total, optionally filtered by a column/value."""
    data = df.copy()
    if group_col and group_val and group_val != "All":
        data = data[data[group_col] == group_val]
    s = data.groupby("Month")["Sales"].sum().sort_index()
    # Ensure a continuous monthly index (fill missing months with 0)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full_idx, fill_value=0.0)
    s.index.name = "Month"
    return s


@st.cache_data
def weekly_series(df: pd.DataFrame, group_col: str | None = None, group_val: str | None = None) -> pd.Series:
    """Aggregate Sales to a weekly total, optionally filtered by a column/value (matches Task 5 methodology)."""
    data = df.copy()
    if group_col and group_val and group_val != "All":
        data = data[data[group_col] == group_val]
    s = data.set_index("Order Date").resample("W")["Sales"].sum()
    s.index.name = "Week"
    return s


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
@st.cache_data
def run_forecast(series: pd.Series, horizon: int, test_size: int = 6):
    """
    Fit Holt-Winters Exponential Smoothing.
    Returns: fitted model forecast for `horizon` months beyond the data,
    plus MAE/RMSE computed on a held-out test window.
    """
    series = series.astype(float)
    n = len(series)
    test_size = min(test_size, max(2, n // 5))

    train, test = series.iloc[: n - test_size], series.iloc[n - test_size :]

    seasonal_periods = 12 if len(train) >= 24 else None
    seasonal = "add" if seasonal_periods else None

    def fit(s):
        try:
            model = ExponentialSmoothing(
                s, trend="add", seasonal=seasonal, seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
            return model
        except Exception:
            model = ExponentialSmoothing(s, trend="add", initialization_method="estimated").fit()
            return model

    # --- evaluate on held-out test window ---
    eval_model = fit(train)
    test_pred = eval_model.forecast(len(test))
    mae = float(np.mean(np.abs(test.values - test_pred.values)))
    rmse = float(np.sqrt(np.mean((test.values - test_pred.values) ** 2)))

    # --- refit on full series, forecast the requested horizon ---
    full_model = fit(series)
    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future_pred = pd.Series(full_model.forecast(horizon).values, index=future_index)

    return future_pred, mae, rmse, test, test_pred


# ---------------------------------------------------------------------------
# Anomaly detection — Z-score based (matches Task 5 methodology: weekly sales,
# flagged where the deviation from the overall mean exceeds a z-score threshold)
# ---------------------------------------------------------------------------
@st.cache_data
def detect_anomalies(series: pd.Series, z_thresh: float = 3.0):
    mean = series.mean()
    std = series.std() if series.std() > 0 else 1.0
    z = (series - mean) / std
    anomalies = series[np.abs(z) > z_thresh]
    return z, anomalies


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
@st.cache_data
def build_subcategory_features(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby(["Sub-Category", "Month"])["Sales"].sum().unstack(fill_value=0)
    monthly = monthly.reindex(sorted(monthly.columns), axis=1)

    features = pd.DataFrame(index=monthly.index)
    features["total_sales"] = monthly.sum(axis=1)
    features["avg_monthly_sales"] = monthly.mean(axis=1)
    features["std_monthly_sales"] = monthly.std(axis=1)
    features["cv"] = (features["std_monthly_sales"] / features["avg_monthly_sales"].replace(0, np.nan)).fillna(0)

    x = np.arange(monthly.shape[1])
    slopes = []
    for _, row in monthly.iterrows():
        y = row.values
        slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        slopes.append(slope)
    features["trend_slope"] = slopes
    return features


@st.cache_data
def compute_elbow(df: pd.DataFrame, k_max: int = 10):
    """WCSS (inertia) for k=1..k_max — reproduces the Task 6 Elbow Method chart."""
    features = build_subcategory_features(df)
    X = StandardScaler().fit_transform(features)
    k_max = min(k_max, len(features))
    wcss = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        wcss.append(km.inertia_)
    return list(range(1, k_max + 1)), wcss


@st.cache_data
def cluster_subcategories(df: pd.DataFrame, k: int = 3):
    features = build_subcategory_features(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    k = min(k, len(features))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    features["Cluster"] = labels

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    features["PC1"] = coords[:, 0]
    features["PC2"] = coords[:, 1]

    return features.reset_index()


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
df = load_data(DATA_PATH)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Superstore Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "1️⃣ Sales Overview",
        "2️⃣ Forecast Explorer",
        "3️⃣ Anomaly Report",
        "4️⃣ Product Demand Segments",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(df):,} orders  \n"
    f"Range: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}"
)

# ===========================================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ===========================================================================
if page.startswith("1"):
    st.title("Sales Overview Dashboard")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        region_filter = st.multiselect(
            "Filter by Region", options=sorted(df["Region"].unique()), default=None
        )
    with col_f2:
        category_filter = st.multiselect(
            "Filter by Category", options=sorted(df["Category"].unique()), default=None
        )

    fdf = df.copy()
    if region_filter:
        fdf = fdf[fdf["Region"].isin(region_filter)]
    if category_filter:
        fdf = fdf[fdf["Category"].isin(category_filter)]

    total_sales = fdf["Sales"].sum()
    total_orders = fdf["Order ID"].nunique()
    avg_order = total_sales / total_orders if total_orders else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales", f"${total_sales:,.0f}")
    m2.metric("Total Orders", f"{total_orders:,}")
    m3.metric("Avg. Order Value", f"${avg_order:,.2f}")

    st.subheader("Total Sales by Year")
    yearly = fdf.groupby("Year")["Sales"].sum().reset_index()
    fig_year = px.bar(yearly, x="Year", y="Sales", text_auto=".2s", color="Sales",
                       color_continuous_scale="Blues")
    fig_year.update_layout(yaxis_title="Total Sales ($)", coloraxis_showscale=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = fdf.groupby("Month")["Sales"].sum().reset_index()
    fig_month = px.line(monthly, x="Month", y="Sales", markers=True)
    fig_month.update_layout(yaxis_title="Sales ($)")
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Sales by Region and Category")
    reg_cat = fdf.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig_reg_cat = px.bar(
        reg_cat, x="Region", y="Sales", color="Category", barmode="group", text_auto=".2s"
    )
    fig_reg_cat.update_layout(yaxis_title="Sales ($)")
    st.plotly_chart(fig_reg_cat, use_container_width=True)

    with st.expander("View underlying data"):
        st.dataframe(fdf.head(500), use_container_width=True)

# ===========================================================================
# PAGE 2 — FORECAST EXPLORER
# ===========================================================================
elif page.startswith("2"):
    st.title("Forecast Explorer")
    st.caption("Model: Holt-Winters Exponential Smoothing (trend + yearly seasonality)")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        dim = st.selectbox("Select dimension", ["Category", "Region"])
    with col2:
        options = ["All"] + sorted(df[dim].unique().tolist())
        value = st.selectbox(f"Select {dim}", options)
    with col3:
        horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    series = monthly_series(df, group_col=dim, group_val=value)

    if len(series) < 12:
        st.warning("Not enough monthly history to forecast reliably for this selection.")
    else:
        future_pred, mae, rmse, test, test_pred = run_forecast(series, horizon=horizon)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="Actual Sales"))
        fig.add_trace(go.Scatter(x=future_pred.index, y=future_pred.values, mode="lines+markers",
                                  name=f"Forecast (+{horizon}mo)", line=dict(dash="dash", color="firebrick")))
        fig.add_trace(go.Scatter(x=test.index, y=test_pred.values, mode="lines", name="Backtest Fit",
                                  line=dict(dash="dot", color="orange")))
        fig.update_layout(
            title=f"Monthly Sales Forecast — {dim}: {value}",
            xaxis_title="Month", yaxis_title="Sales ($)", legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Values")
        st.dataframe(
            future_pred.rename("Forecasted Sales ($)").round(2).reset_index().rename(columns={"index": "Month"}),
            use_container_width=True,
        )

        st.subheader("Model Performance (Backtest)")
        c1, c2 = st.columns(2)
        c1.metric("MAE", f"${mae:,.2f}")
        c2.metric("RMSE", f"${rmse:,.2f}")
        st.caption(
            "MAE / RMSE computed by holding out the last few months of actual data, "
            "refitting the model on the rest, and comparing predictions to the held-out actuals."
        )

# ===========================================================================
# PAGE 3 — ANOMALY REPORT
# ===========================================================================
elif page.startswith("3"):
    st.title("Anomaly Report")
    st.caption("Z-Score Based Anomaly Detection on weekly sales (Task 5 methodology)")

    col1, col2 = st.columns([1, 1])
    with col1:
        dim = st.selectbox("Scope", ["Overall", "Category", "Region"])
    value = "All"
    if dim != "Overall":
        with col2:
            value = st.selectbox(f"Select {dim}", ["All"] + sorted(df[dim].unique().tolist()))
    group_col = None if dim == "Overall" else dim

    z_thresh = st.slider("Anomaly sensitivity (z-score threshold)", 1.5, 4.0, 3.0, 0.1)

    series = weekly_series(df, group_col=group_col, group_val=value)
    z, anomalies = detect_anomalies(series, z_thresh=z_thresh)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Weekly Sales",
                              line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies.values, mode="markers", name="Z-Score Anomalies",
        marker=dict(color="green", size=11, symbol="circle"),
    ))
    fig.update_layout(title="Z-Score Based Anomaly Detection", xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomaly Dates")
    if anomalies.empty:
        st.info("No anomalies detected at the current sensitivity threshold.")
    else:
        anomaly_table = anomalies.rename("Sales ($)").round(2).reset_index().rename(columns={"index": "Week"})
        anomaly_table["Z-score"] = z.loc[anomalies.index].round(2).values
        st.dataframe(anomaly_table.sort_values("Week"), use_container_width=True)

# ===========================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ===========================================================================
elif page.startswith("4"):
    st.title("Product Demand Segments")
    st.caption("KMeans clustering of Sub-Categories based on demand behavior (Task 6 methodology)")

    st.subheader("Elbow Method")
    k_choices, wcss = compute_elbow(df, k_max=10)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=k_choices, y=wcss, mode="lines+markers",
                                    line=dict(color="#1f77b4"), marker=dict(size=8)))
    fig_elbow.update_layout(title="Elbow Method", xaxis_title="Number of Clusters", yaxis_title="WCSS")
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.caption("WCSS (within-cluster sum of squares) drops sharply then levels off — the 'elbow' marks a good choice of k.")

    k = st.slider("Number of clusters (k)", 2, 6, 3)
    clusters = cluster_subcategories(df, k=k)

    st.subheader("Cluster Map (PCA projection)")
    fig = px.scatter(
        clusters, x="PC1", y="PC2", color=clusters["Cluster"].astype(str),
        text="Sub-Category", size="total_sales", hover_data=["total_sales", "avg_monthly_sales", "trend_slope"],
        labels={"color": "Cluster"},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category → Cluster Assignment")
    display_cols = ["Sub-Category", "Cluster", "total_sales", "avg_monthly_sales", "std_monthly_sales", "cv", "trend_slope"]
    show = clusters[display_cols].copy()
    show.columns = ["Sub-Category", "Cluster", "Total Sales ($)", "Avg Monthly Sales ($)",
                     "Std Monthly Sales", "Coefficient of Variation", "Trend Slope"]
    for c in ["Total Sales ($)", "Avg Monthly Sales ($)", "Std Monthly Sales", "Trend Slope"]:
        show[c] = show[c].round(2)
    show["Coefficient of Variation"] = show["Coefficient of Variation"].round(3)
    st.dataframe(show.sort_values(["Cluster", "Total Sales ($)"], ascending=[True, False]),
                 use_container_width=True)

    with st.expander("What do the clusters mean?"):
        st.markdown(
            "- **total_sales / avg_monthly_sales**: overall demand size\n"
            "- **std_monthly_sales / cv**: volatility of monthly demand\n"
            "- **trend_slope**: whether demand is growing, flat, or declining over time\n\n"
            "Sub-categories grouped together have similar demand magnitude, volatility, and trend — "
            "useful for inventory planning, promotions, and forecasting model selection."
        )
