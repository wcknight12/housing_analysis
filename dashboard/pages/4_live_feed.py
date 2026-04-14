"""
Page 4 — Live Feed (Kafka Structured Streaming)
================================================
Simulates Kafka mortgage-rate and listing alerts in the browser.
In production, connect to a running Kafka broker (see docker-compose.yml).
"""

import os
import sys
import time
import random
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from streaming.producer import _mortgage_rate_message, _new_listing_message

st.set_page_config(page_title="Live Feed", page_icon="📡", layout="wide")
st.title("📡 Live Mortgage Rates & New Listing Alerts")
st.caption(
    "Simulated Kafka stream — in production, wire to a live Kafka broker "
    "(see docker-compose.yml). Refresh or use the auto-refresh button below."
)

# ---------------------------------------------------------------------------
# Session-state history buffers
# ---------------------------------------------------------------------------

if "rate_history" not in st.session_state:
    # seed with 20 historical points
    seed_rate = 6.85
    history = []
    for i in range(20):
        seed_rate += random.gauss(0, 0.04)
        seed_rate = round(max(3.0, min(10.0, seed_rate)), 3)
        history.append({
            "time": i,
            "rate_30yr": seed_rate,
            "rate_15yr": round(seed_rate - random.uniform(0.5, 0.9), 3),
        })
    st.session_state["rate_history"] = history

if "listing_history" not in st.session_state:
    st.session_state["listing_history"] = []

# ---------------------------------------------------------------------------
# Refresh controls
# ---------------------------------------------------------------------------

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 4])
with col_ctrl1:
    if st.button("🔄 Fetch new events"):
        # Add new rate point
        msg = _mortgage_rate_message()
        history = st.session_state["rate_history"]
        history.append({
            "time": len(history),
            "rate_30yr": msg["rate_30yr_fixed"],
            "rate_15yr": msg["rate_15yr_fixed"],
        })
        st.session_state["rate_history"] = history[-50:]  # keep last 50

        # Add new listings
        for _ in range(random.randint(1, 4)):
            lst = _new_listing_message()
            st.session_state["listing_history"].insert(0, lst)
        st.session_state["listing_history"] = st.session_state["listing_history"][:30]

with col_ctrl2:
    auto_refresh = st.checkbox("Auto-refresh (5 s)", value=False)

if auto_refresh:
    time.sleep(5)
    st.rerun()

# ---------------------------------------------------------------------------
# Current mortgage rates
# ---------------------------------------------------------------------------

st.subheader("Current Mortgage Rates")

latest_rate = st.session_state["rate_history"][-1]
r1, r2, r3 = st.columns(3)
r1.metric("30-Year Fixed", f"{latest_rate['rate_30yr']:.3f}%",
          delta=f"{latest_rate['rate_30yr'] - st.session_state['rate_history'][-2]['rate_30yr']:.3f}%"
          if len(st.session_state["rate_history"]) > 1 else None)
r2.metric("15-Year Fixed", f"{latest_rate['rate_15yr']:.3f}%")
r3.metric("Events Received", len(st.session_state["rate_history"]))

# Sparkline
rate_df = pd.DataFrame(st.session_state["rate_history"])
fig_rate = go.Figure()
fig_rate.add_trace(go.Scatter(x=rate_df["time"], y=rate_df["rate_30yr"],
                               name="30-yr Fixed", line=dict(color="#1f77b4")))
fig_rate.add_trace(go.Scatter(x=rate_df["time"], y=rate_df["rate_15yr"],
                               name="15-yr Fixed", line=dict(color="#ff7f0e", dash="dot")))
fig_rate.update_layout(
    title="Mortgage Rate Stream (simulated Kafka)", height=280,
    xaxis_title="Event #", yaxis_title="Rate (%)",
    legend_title_text="Product",
    yaxis=dict(range=[
        min(rate_df["rate_15yr"].min(), rate_df["rate_30yr"].min()) - 0.2,
        max(rate_df["rate_30yr"].max(), rate_df["rate_15yr"].max()) + 0.2,
    ]),
)
st.plotly_chart(fig_rate, use_container_width=True)

# ---------------------------------------------------------------------------
# New listing alerts table
# ---------------------------------------------------------------------------

st.subheader("New Listing Alerts")

if st.session_state["listing_history"]:
    listing_df = pd.DataFrame(st.session_state["listing_history"])
    listing_df["price"] = listing_df["price"].apply(lambda x: f"${x:,.0f}")
    listing_df["price_per_sqft"] = listing_df["price_per_sqft"].apply(lambda x: f"${x:.0f}/sqft")
    st.dataframe(
        listing_df[["timestamp", "listing_id", "city", "zip_code",
                    "price", "sqft", "bedrooms", "bathrooms", "price_per_sqft"]]
        .rename(columns={"timestamp": "Time", "listing_id": "ID",
                         "city": "City", "zip_code": "ZIP", "sqft": "Sq Ft",
                         "bedrooms": "Beds", "bathrooms": "Baths"}),
        use_container_width=True, height=350,
    )

    # Summary by city
    st.subheader("Live Listings — City Summary")
    raw_listing_df = pd.DataFrame(st.session_state["listing_history"])
    city_counts = raw_listing_df.groupby("city").agg(
        count=("listing_id", "count"),
        avg_price=("price", "mean"),
    ).reset_index()
    fig_cities = px.bar(
        city_counts, x="city", y="count", color="city",
        title="New Listings Received by City (current session)",
        labels={"count": "Listing Count", "city": "City"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_cities.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_cities, use_container_width=True)
else:
    st.info("Press **Fetch new events** to receive listing alerts from the simulated Kafka stream.")

st.divider()
st.markdown(
    """
    **Production setup:** start the full Kafka stack with:
    ```bash
    docker-compose up -d
    python -m streaming.producer --mode all   # in a separate terminal
    python -m streaming.consumer --topic all  # in a separate terminal
    ```
    """
)
