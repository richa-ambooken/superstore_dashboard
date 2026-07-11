import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Sales Overview | Superstore Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    yearly = pd.read_csv("artifacts/yearly_sales.csv")
    monthly = pd.read_csv("artifacts/monthly_sales.csv", parse_dates=["Month"])
    region_cat = pd.read_csv("artifacts/region_category_sales.csv")
    monthly_region = pd.read_csv(
        "artifacts/monthly_sales_by_region.csv", parse_dates=["Month"]
    )
    monthly_category = pd.read_csv(
        "artifacts/monthly_sales_by_category.csv", parse_dates=["Month"]
    )
    return yearly, monthly, region_cat, monthly_region, monthly_category


yearly, monthly, region_cat, monthly_region, monthly_category = load_data()

st.title("📊 Sales Overview Dashboard")
st.caption("Superstore sales data · 2015 – 2018")

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${monthly['Total_Sales'].sum():,.0f}")
col2.metric("Years Covered", f"{yearly['Year'].min()} – {yearly['Year'].max()}")
col3.metric("Regions", region_cat["Region"].nunique())
col4.metric("Categories", region_cat["Category"].nunique())

st.divider()

# ---- Total sales by year ----
st.subheader("Total Sales by Year")
fig_year = px.bar(
    yearly,
    x="Year",
    y="Total_Sales",
    text_auto=".2s",
    color="Total_Sales",
    color_continuous_scale="Blues",
)
fig_year.update_layout(showlegend=False, yaxis_title="Total Sales ($)")
st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# ---- Monthly sales trend ----
st.subheader("Monthly Sales Trend")
fig_month = px.line(
    monthly, x="Month", y="Total_Sales", markers=True
)
fig_month.update_layout(yaxis_title="Total Sales ($)", xaxis_title="Month")
st.plotly_chart(fig_month, use_container_width=True)

st.divider()

# ---- Sales by region & category with interactive filters ----
st.subheader("Sales by Region and Category")

f1, f2 = st.columns(2)
with f1:
    regions = st.multiselect(
        "Filter by Region",
        options=sorted(region_cat["Region"].unique()),
        default=sorted(region_cat["Region"].unique()),
    )
with f2:
    categories = st.multiselect(
        "Filter by Category",
        options=sorted(region_cat["Category"].unique()),
        default=sorted(region_cat["Category"].unique()),
    )

filtered = region_cat[
    region_cat["Region"].isin(regions) & region_cat["Category"].isin(categories)
]

c1, c2 = st.columns(2)
with c1:
    by_region_cat = filtered.groupby(["Region", "Category"], as_index=False)["Sales"].sum()
    fig_rc = px.bar(
        by_region_cat,
        x="Region",
        y="Sales",
        color="Category",
        barmode="group",
        title="Sales by Region and Category",
    )
    st.plotly_chart(fig_rc, use_container_width=True)

with c2:
    by_subcat = (
        filtered.groupby("Sub-Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    fig_sc = px.bar(
        by_subcat,
        x="Sales",
        y="Sub-Category",
        orientation="h",
        title="Sales by Sub-Category",
        color="Sales",
        color_continuous_scale="Teal",
    )
    fig_sc.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)

with st.expander("View filtered data table"):
    st.dataframe(filtered.sort_values("Sales", ascending=False), use_container_width=True)

st.sidebar.success("Use the pages above to explore Forecasts, Anomalies, and Demand Segments.")
