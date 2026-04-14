"""
Page 6 — Comparable (Comp) Analyzer
======================================
Find similar listings by budget, features, and location to
help buyers benchmark what they should pay for a home.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Comp Analyzer", page_icon="🔍", layout="wide")
st.title("🔍 Comparable Home Analyzer")
st.caption("Find and benchmark comparable listings by budget, size, location, and features")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_listings():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "zillow_tx_listings.csv")
    df = pd.read_csv(path, dtype={"zip_code": str})
    df.dropna(subset=["price", "sqft"], inplace=True)
    return df

listings = load_listings()

# ---------------------------------------------------------------------------
# Sidebar — subject property
# ---------------------------------------------------------------------------

st.sidebar.header("🏠 Subject Property")
st.sidebar.caption("Describe the home you're considering. Comps will be pulled within ±20% of these specs.")

city_options = sorted(listings["city"].unique())
sub_city = st.sidebar.selectbox("City", city_options, index=0)
sub_budget = st.sidebar.number_input("Your Budget ($)", 100_000, 2_000_000, 400_000, step=10_000)
sub_sqft = st.sidebar.slider("Target Sq Ft", 800, 5_000, 1_800, step=100)
sub_beds = st.sidebar.selectbox("Min Bedrooms", [2, 3, 4, 5], index=1)
sub_baths = st.sidebar.selectbox("Min Bathrooms", [1.5, 2.0, 2.5, 3.0, 3.5], index=1)
sub_school = st.sidebar.slider("Min School Rating", 1.0, 10.0, 6.0, step=0.5)

st.sidebar.divider()
st.sidebar.header("📐 Comp Tolerance")
price_tol = st.sidebar.slider("Price tolerance (%)", 5, 40, 20, step=5)
sqft_tol = st.sidebar.slider("Sqft tolerance (%)", 5, 40, 20, step=5)

# ---------------------------------------------------------------------------
# Filter comps
# ---------------------------------------------------------------------------

price_lo = sub_budget * (1 - price_tol / 100)
price_hi = sub_budget * (1 + price_tol / 100)
sqft_lo = sub_sqft * (1 - sqft_tol / 100)
sqft_hi = sub_sqft * (1 + sqft_tol / 100)

comps = listings[
    (listings["city"] == sub_city)
    & listings["price"].between(price_lo, price_hi)
    & listings["sqft"].between(sqft_lo, sqft_hi)
    & (listings["bedrooms"] >= sub_beds)
    & (listings["bathrooms"] >= sub_baths)
    & (listings["school_rating"] >= sub_school)
].copy()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.subheader(f"Comps in {sub_city} matching your criteria")

if comps.empty:
    st.warning("No comparables found. Try relaxing your search criteria using the sidebar sliders.")
    st.stop()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Comps Found", len(comps))
m2.metric("Median Price", f"${comps['price'].median():,.0f}")
m3.metric("Median $/sqft", f"${comps['price_per_sqft'].median():,.2f}")
m4.metric("Median Sq Ft", f"{comps['sqft'].median():,.0f}")
m5.metric("Avg School Rating", f"{comps['school_rating'].mean():.1f}")

# ---------------------------------------------------------------------------
# Price vs sqft scatter of comps
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Price vs. Square Footage")
    fig1 = px.scatter(
        comps, x="sqft", y="price",
        color="school_rating",
        size="bedrooms",
        hover_data=["zip_code", "bathrooms", "age_years", "price_per_sqft"],
        color_continuous_scale="YlOrRd",
        labels={"sqft": "Sq Ft", "price": "Price ($)", "school_rating": "School"},
        title=f"{sub_city} — Comps Scatter",
    )
    # Mark the target
    fig1.add_scatter(
        x=[sub_sqft], y=[sub_budget],
        mode="markers",
        marker=dict(color="blue", size=16, symbol="star"),
        name="Your Target",
    )
    fig1.update_layout(yaxis_tickformat="$,.0f", height=380)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("Price Distribution of Comps")
    fig2 = px.histogram(
        comps, x="price",
        nbins=20,
        title="Price Histogram",
        labels={"price": "Price ($)"},
        color_discrete_sequence=["steelblue"],
    )
    fig2.add_vline(x=sub_budget, line_dash="dash", line_color="red",
                   annotation_text="Your budget", annotation_position="top right")
    fig2.add_vline(x=comps["price"].median(), line_dash="dot", line_color="green",
                   annotation_text="Comp median", annotation_position="top left")
    fig2.update_layout(xaxis_tickformat="$,.0f", height=380)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# $/sqft comparison
# ---------------------------------------------------------------------------

st.subheader("Price-per-Sq-Ft — Your Budget vs. Comps")

your_ppsf = sub_budget / sub_sqft
comp_median_ppsf = comps["price_per_sqft"].median()
delta_pct = (your_ppsf - comp_median_ppsf) / comp_median_ppsf * 100

c1, c2, c3 = st.columns(3)
c1.metric("Your $/sqft", f"${your_ppsf:.2f}")
c2.metric("Comp Median $/sqft", f"${comp_median_ppsf:.2f}")
c3.metric(
    "vs. Market",
    f"{delta_pct:+.1f}%",
    delta=f"{'Paying a premium' if delta_pct > 5 else 'Good value' if delta_pct < -5 else 'At market'}",
    delta_color="inverse",
)

# ---------------------------------------------------------------------------
# Best-value comps table
# ---------------------------------------------------------------------------

st.subheader("Top Comps by Value Score")

comps["value_score"] = (
    (comps["school_rating"] / 10)
    - (comps["price"] / sub_budget)
    + (comps["sqft"] / sub_sqft * 0.5)
).round(3)

top_comps = comps.sort_values("value_score", ascending=False).head(15)

display_cols = ["zip_code", "price", "sqft", "price_per_sqft",
                "bedrooms", "bathrooms", "school_rating", "age_years", "value_score"]
display_labels = {
    "zip_code": "ZIP", "price": "Price", "sqft": "Sq Ft",
    "price_per_sqft": "$/sqft", "bedrooms": "Beds",
    "bathrooms": "Baths", "school_rating": "School",
    "age_years": "Age", "value_score": "Value Score",
}

st.dataframe(
    top_comps[display_cols]
    .rename(columns=display_labels)
    .style
    .background_gradient(subset=["Value Score"], cmap="YlGn")
    .format({
        "Price": "${:,.0f}",
        "$/sqft": "${:.2f}",
        "School": "{:.1f}",
        "Value Score": "{:.3f}",
    }),
    use_container_width=True,
    height=400,
)

# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------

st.divider()
if delta_pct > 10:
    st.warning(
        f"⚠️  Your budget of **${sub_budget:,.0f}** implies **${your_ppsf:.0f}/sqft** "
        f"— that's **{delta_pct:.1f}% above** the comp median of **${comp_median_ppsf:.0f}/sqft**. "
        f"Consider negotiating or looking at slightly smaller homes."
    )
elif delta_pct < -10:
    st.success(
        f"✅  Your budget implies **${your_ppsf:.0f}/sqft** — **{abs(delta_pct):.1f}% below** "
        f"the comp median. You have room to negotiate up on features like school rating or lot size."
    )
else:
    st.info(
        f"ℹ️  Your budget is priced **at market** ({delta_pct:+.1f}% vs. comps). "
        f"Focus on school rating and age of home to differentiate value."
    )
