import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Product Demand Segments", page_icon="🧩", layout="wide")


@st.cache_data
def load_clusters():
    return pd.read_csv("artifacts/clusters.csv")


df = load_clusters()

st.title("🧩 Product Demand Segments")
st.caption(
    "Sub-Categories are clustered (K-Means, k=3) using total sales, average monthly "
    "sales, volatility, and growth trend — then labelled by demand level."
)

segment_colors = {
    "High Demand": "#2ECC71",
    "Medium Demand": "#F1C40F",
    "Low Demand": "#E74C3C",
}

# ---- summary ----
seg_counts = df["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]
cols = st.columns(len(seg_counts))
for c, (_, row) in zip(cols, seg_counts.iterrows()):
    c.metric(row["Segment"], f"{row['Count']} sub-categories")

st.divider()

# ---- cluster chart ----
st.subheader("Cluster Chart: Total Sales vs. Average Monthly Sales")
fig = px.scatter(
    df,
    x="Total_Sales",
    y="Avg_Monthly_Sales",
    color="Segment",
    size="Volatility",
    hover_name="Sub-Category",
    color_discrete_map=segment_colors,
    labels={
        "Total_Sales": "Total Sales ($)",
        "Avg_Monthly_Sales": "Average Monthly Sales ($)",
    },
    text="Sub-Category",
)
fig.update_traces(textposition="top center")
fig.update_layout(legend_title_text="Demand Segment")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- growth vs volatility view ----
st.subheader("Growth vs. Volatility by Sub-Category")
fig2 = px.scatter(
    df,
    x="Growth",
    y="Volatility",
    color="Segment",
    hover_name="Sub-Category",
    color_discrete_map=segment_colors,
    labels={"Growth": "Growth Trend (2nd half vs 1st half)", "Volatility": "Volatility (CV)"},
    text="Sub-Category",
)
fig2.update_traces(textposition="top center")
fig2.add_vline(x=0, line_dash="dot", line_color="gray")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---- table ----
st.subheader("Sub-Category → Demand Cluster Mapping")

filter_segment = st.multiselect(
    "Filter by segment", options=sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique())
)
table = df[df["Segment"].isin(filter_segment)].copy()
table = table[["Sub-Category", "Segment", "Total_Sales", "Avg_Monthly_Sales", "Volatility", "Growth"]]
table["Total_Sales"] = table["Total_Sales"].map(lambda v: f"${v:,.0f}")
table["Avg_Monthly_Sales"] = table["Avg_Monthly_Sales"].map(lambda v: f"${v:,.0f}")
table["Volatility"] = table["Volatility"].round(2)
table["Growth"] = table["Growth"].map(lambda v: f"{v*100:,.1f}%")
table = table.sort_values("Segment")

st.dataframe(table, use_container_width=True, hide_index=True)
