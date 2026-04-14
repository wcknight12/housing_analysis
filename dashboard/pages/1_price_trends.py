"""
Page 1 — Price Trends
=====================
Zillow ZHVI batch ETL results visualised as interactive trend lines,
YoY change maps, and ZIP-level heat tables.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Price Trends", page_icon="📈", layout="wide")
st.title("📈 Texas Home Price Trends")
st.caption("Source: Zillow ZHVI (simulated) — processed via PySpark ETL pipeline")


# ---------------------------------------------------------------------------
# Load ZHVI data (pandas for dashboard speed; PySpark ETL runs separately)
# ---------------------------------------------------------------------------

@st.cache_data
def load_zhvi():
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "zillow_tx_zhvi.csv")
    df = pd.read_csv(data_path, dtype={"RegionName": str})
    meta_cols = ["RegionID", "SizeRank", "RegionName", "RegionType",
                 "StateName", "State", "City", "Metro", "CountyName"]
    date_cols = [c for c in df.columns if c not in meta_cols and c[0].isdigit()]

    # Melt wide → long
    long_df = df.melt(
        id_vars=["RegionName", "City", "Metro", "State"],
        value_vars=date_cols,
        var_name="date",
        value_name="zhvi",
    ).rename(columns={"RegionName": "zip_code", "City": "city", "Metro": "metro"})
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df.dropna(subset=["zhvi"], inplace=True)
    return long_df


df = load_zhvi()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

metros = sorted(df["metro"].dropna().unique())
selected_metros = st.sidebar.multiselect("Metro area", metros, default=metros)

min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)

filtered = df[
    df["metro"].isin(selected_metros)
    & (df["date"] >= pd.Timestamp(start_date))
    & (df["date"] <= pd.Timestamp(end_date))
]

# ---------------------------------------------------------------------------
# Section 1: Median ZHVI trend by metro
# ---------------------------------------------------------------------------

st.subheader("Median Home Value Trend by Metro")

monthly_metro = (
    filtered.groupby(["metro", "date"])["zhvi"]
    .median()
    .reset_index()
    .rename(columns={"zhvi": "median_zhvi"})
)

fig1 = px.line(
    monthly_metro,
    x="date",
    y="median_zhvi",
    color="metro",
    labels={"median_zhvi": "Median ZHVI ($)", "date": ""},
)
fig1.update_layout(yaxis_tickformat="$,.0f", hovermode="x unified", height=400)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Latest median ZHVI per city
# ---------------------------------------------------------------------------

st.subheader("Latest Median Home Value by City")

latest = filtered[filtered["date"] == filtered["date"].max()]
city_latest = (
    latest.groupby("city")["zhvi"].median().reset_index()
    .sort_values("zhvi", ascending=False)
)

fig2 = px.bar(
    city_latest,
    x="city",
    y="zhvi",
    color="city",
    labels={"zhvi": "Median ZHVI ($)", "city": "City"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig2.update_layout(yaxis_tickformat="$,.0f", showlegend=False, height=350)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Year-over-Year change
# ---------------------------------------------------------------------------

st.subheader("Year-over-Year Price Change (%) by ZIP")

# Compute YoY for each ZIP
yoy_df = filtered.sort_values("date")
yoy_df = yoy_df.merge(
    yoy_df.assign(date_lag=yoy_df["date"] + pd.DateOffset(months=12))[
        ["zip_code", "date_lag", "zhvi"]
    ].rename(columns={"date_lag": "date", "zhvi": "zhvi_lag"}),
    on=["zip_code", "date"],
    how="left",
)
yoy_df["yoy_pct"] = ((yoy_df["zhvi"] - yoy_df["zhvi_lag"]) / yoy_df["zhvi_lag"] * 100).round(2)
yoy_latest = yoy_df[yoy_df["date"] == yoy_df["date"].max()].dropna(subset=["yoy_pct"])

fig3 = px.bar(
    yoy_latest.sort_values("yoy_pct", ascending=False).head(30),
    x="zip_code",
    y="yoy_pct",
    color="city",
    title="Top 30 ZIPs — Year-over-Year Price Change (%)",
    labels={"yoy_pct": "YoY Change (%)", "zip_code": "ZIP Code"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Data table
# ---------------------------------------------------------------------------

with st.expander("📊 ZIP-level data table"):
    show = city_latest.rename(columns={"zhvi": "Median ZHVI", "city": "City"})
    st.dataframe(
        show.style.format({"Median ZHVI": "${:,.0f}"}),
        use_container_width=True,
    )
