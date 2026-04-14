"""
Page 3 — Neighborhood Livability Scorer
========================================
User-adjustable weight sliders → composite livability score per ZIP,
joined with Census ACS data.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neighborhood.census import load_census_pandas
from neighborhood.scorer import (
    ScorerWeights,
    WEIGHT_LABELS,
    compute_composite_score,
    rank_zips,
)

st.set_page_config(page_title="Neighborhood Score", page_icon="🏘️", layout="wide")
st.title("🏘️ Neighborhood Livability Scorer")
st.caption("Composite score built from Census ACS data — adjust weights to match your priorities")

# ---------------------------------------------------------------------------
# Load Census data
# ---------------------------------------------------------------------------

@st.cache_data
def load_census():
    return load_census_pandas()

census = load_census()

# ---------------------------------------------------------------------------
# Sidebar — weight sliders
# ---------------------------------------------------------------------------

st.sidebar.header("⚖️ Adjust Your Priorities")
st.sidebar.caption("Drag sliders to weight what matters most to you. Weights are auto-normalized.")

raw_weights = {}
for key, label in WEIGHT_LABELS.items():
    raw_weights[key] = st.sidebar.slider(label, 0, 10, 5)

total = sum(raw_weights.values()) or 1
weights = ScorerWeights(**{k: v / total for k, v in raw_weights.items()})

st.sidebar.divider()
st.sidebar.subheader("Filter by Metro")
metros = sorted(census["metro"].dropna().unique())
selected_metros = st.sidebar.multiselect("Metro", metros, default=metros)

# ---------------------------------------------------------------------------
# Compute scores
# ---------------------------------------------------------------------------

filtered_census = census[census["metro"].isin(selected_metros)]
scored = compute_composite_score(filtered_census, weights)

# ---------------------------------------------------------------------------
# Top 10 ZIPs
# ---------------------------------------------------------------------------

st.subheader("Top ZIP Codes by Livability Score")

top10 = scored.head(10)
cols_display = [
    "zip_code", "city_name", "metro", "livability_score", "livability_tier",
    "school_quality_score", "affordability_score", "safety_score",
    "walkability_score", "economic_health_score", "commute_score",
]
st.dataframe(
    top10[cols_display].style.background_gradient(subset=["livability_score"], cmap="YlGn")
        .format({c: "{:.1f}" for c in cols_display if "score" in c}),
    use_container_width=True,
    height=350,
)

# ---------------------------------------------------------------------------
# Score distribution map
# ---------------------------------------------------------------------------

st.subheader("Livability Score Distribution — All ZIPs")

fig_bar = px.bar(
    scored,
    x="zip_code",
    y="livability_score",
    color="livability_tier",
    color_discrete_map={
        "Excellent": "#2ca02c",
        "Good": "#1f77b4",
        "Average": "#ff7f0e",
        "Developing": "#d62728",
    },
    labels={"livability_score": "Livability Score (0–100)", "zip_code": "ZIP"},
    title="Livability Scores by ZIP Code",
    category_orders={"livability_tier": ["Excellent", "Good", "Average", "Developing"]},
)
fig_bar.update_layout(height=400)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Radar chart for selected ZIP
# ---------------------------------------------------------------------------

st.subheader("ZIP Livability Radar")

selected_zip = st.selectbox(
    "Select a ZIP to inspect",
    options=scored["zip_code"].tolist(),
    index=0,
)

row = scored[scored["zip_code"] == selected_zip].iloc[0]

radar_dims = {
    "School Quality": row["school_quality_score"],
    "Affordability": row["affordability_score"],
    "Safety": row["safety_score"],
    "Walkability": row["walkability_score"],
    "Economic Health": row["economic_health_score"],
    "Commute": row["commute_score"],
    "Environment": row["environment_score"],
}
categories = list(radar_dims.keys()) + [list(radar_dims.keys())[0]]
values = list(radar_dims.values()) + [list(radar_dims.values())[0]]

fig_radar = go.Figure(
    go.Scatterpolar(r=values, theta=categories, fill="toself", line_color="#1f77b4")
)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    title=f"Livability Radar — ZIP {selected_zip} ({row['city_name']})"
        f" | Score: {row['livability_score']:.1f} / 100",
    height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Sub-score comparison across metros
# ---------------------------------------------------------------------------

st.subheader("Metro-level Sub-score Comparison")

metro_avg = (
    scored.groupby("metro")[
        ["school_quality_score", "affordability_score", "safety_score",
         "walkability_score", "economic_health_score", "commute_score", "environment_score"]
    ].mean().round(1).reset_index()
)

fig_heat = px.imshow(
    metro_avg.set_index("metro"),
    color_continuous_scale="YlGn",
    title="Average Sub-score by Metro",
    labels=dict(color="Score (0–100)"),
    zmin=0, zmax=100,
    text_auto=".1f",
)
fig_heat.update_layout(height=300)
st.plotly_chart(fig_heat, use_container_width=True)
