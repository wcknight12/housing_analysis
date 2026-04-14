"""
Texas Housing Market Analysis — Streamlit Dashboard
====================================================

Multi-page app. Run with:
    streamlit run dashboard/app.py
"""

import os
import sys

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Texas Housing Market Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

st.title("🏠 Texas Housing Market Analysis")
st.markdown(
    """
    **A comprehensive data platform for Texas single-family home buyers and investors.**

    Use the sidebar to navigate between analysis modules:

    | Page | What it shows |
    |---|---|
    | 📈 **Price Trends** | ZHVI trends by metro & ZIP — powered by PySpark ETL |
    | 🤖 **ML Predictions** | GBT price predictions + fair-value scoring per ZIP |
    | 🏘️ **Neighborhood Score** | Livability composite scorer with adjustable weights |
    | 📡 **Live Feed** | Simulated live mortgage rates & new listing alerts |
    | ⚖️ **Buy vs. Wait** | Financial simulator: buy now vs. waiting N years |
    | 🔍 **Comp Analyzer** | Find comparable listings by budget & features |
    """
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Texas Metros Covered", "5")
with col2:
    st.metric("ZIP Codes Analysed", "45+")
with col3:
    st.metric("Years of ZHVI Data", "5+")
with col4:
    st.metric("ML Models", "3")
with col5:
    st.metric("Livability Dimensions", "7")

st.divider()
st.markdown(
    """
    ### Quick Key Findings
    - 🏡 **Square footage** is the #1 price driver across all Texas metros
    - 🎓 Each +1 school rating point ≈ 2–3% price premium
    - 💰 **San Antonio** offers the best affordability for a $85k household
    - 🏙️ **Austin** has the highest median prices and strongest school ratings
    - 📉 Homes 5–10 miles closer to city centre add 10–15% to value
    """
)
