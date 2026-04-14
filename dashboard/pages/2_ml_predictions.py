"""
Page 2 — ML Price Predictions & Fair-Value Scoring
===================================================
Trains a scikit-learn GBT model (PySpark MLlib version available via ETL pipeline)
and visualises predicted vs. actual prices plus per-ZIP fair-value scores.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")
st.title("🤖 ML Price Predictions & Fair-Value Scoring")
st.caption("Gradient Boosted Tree model trained on listing features — ZIP-level fair-value scoring")


# ---------------------------------------------------------------------------
# Load listings
# ---------------------------------------------------------------------------

@st.cache_data
def load_listings():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "zillow_tx_listings.csv")
    df = pd.read_csv(path, dtype={"zip_code": str})
    df.dropna(subset=["price", "sqft"], inplace=True)
    return df


@st.cache_resource
def train_gbt(df):
    features = ["sqft", "bedrooms", "bathrooms", "lot_size_acres",
                 "age_years", "school_rating", "distance_from_center_miles",
                 "property_tax_rate"]
    X = df[features].fillna(df[features].median())
    y = np.log(df["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)
    metrics = {
        "R²": round(r2_score(y_true, y_pred), 3),
        "MAE": f"${mean_absolute_error(y_true, y_pred):,.0f}",
    }
    return model, X_test, y_true, y_pred, metrics, features


listings = load_listings()
model, X_test, y_true, y_pred, metrics, features = train_gbt(listings)

# ---------------------------------------------------------------------------
# Sidebar — predict a custom home
# ---------------------------------------------------------------------------

st.sidebar.header("🔮 Predict a Home's Price")
sqft_in = st.sidebar.slider("Square Footage", 800, 5000, 1800)
beds_in = st.sidebar.selectbox("Bedrooms", [2, 3, 4, 5], index=1)
baths_in = st.sidebar.selectbox("Bathrooms", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], index=2)
lot_in = st.sidebar.slider("Lot Size (acres)", 0.05, 3.0, 0.18, step=0.05)
age_in = st.sidebar.slider("Age of Home (years)", 0, 60, 15)
school_in = st.sidebar.slider("School Rating (1–10)", 1.0, 10.0, 7.5, step=0.1)
dist_in = st.sidebar.slider("Distance from Center (miles)", 1.0, 50.0, 15.0, step=0.5)
tax_in = st.sidebar.slider("Property Tax Rate", 0.015, 0.030, 0.022, step=0.001, format="%.3f")

custom_input = [[sqft_in, beds_in, baths_in, lot_in, age_in, school_in, dist_in, tax_in]]
custom_pred = int(np.exp(model.predict(custom_input)[0]))
st.sidebar.metric("🏷️ Predicted Price", f"${custom_pred:,.0f}")

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------

st.subheader("Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", metrics["R²"])
c2.metric("Mean Abs. Error", metrics["MAE"])
c3.metric("Training Samples", f"{len(listings):,}")

# ---------------------------------------------------------------------------
# Predicted vs Actual
# ---------------------------------------------------------------------------

st.subheader("Predicted vs. Actual Prices")
pred_df = pd.DataFrame({"actual": y_true.values, "predicted": y_pred})
fig1 = px.scatter(
    pred_df, x="actual", y="predicted",
    labels={"actual": "Actual Price ($)", "predicted": "Predicted Price ($)"},
    opacity=0.4,
    trendline="ols",
)
fig1.add_shape(type="line", x0=pred_df["actual"].min(), x1=pred_df["actual"].max(),
               y0=pred_df["actual"].min(), y1=pred_df["actual"].max(),
               line=dict(color="red", dash="dash"))
fig1.update_layout(xaxis_tickformat="$,.0f", yaxis_tickformat="$,.0f", height=420)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Feature Importances")
    labels = {
        "sqft": "Square Footage", "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms", "lot_size_acres": "Lot Size",
        "age_years": "Age of Home", "school_rating": "School Rating",
        "distance_from_center_miles": "Distance from Center",
        "property_tax_rate": "Property Tax Rate",
    }
    imp_df = pd.DataFrame({
        "feature": [labels.get(f, f) for f in features],
        "importance": model.feature_importances_,
    }).sort_values("importance")
    fig2 = px.bar(imp_df, x="importance", y="feature", orientation="h",
                  color="importance", color_continuous_scale="Blues")
    fig2.update_layout(coloraxis_showscale=False, height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Fair-value scoring per ZIP
# ---------------------------------------------------------------------------

with col_b:
    st.subheader("Fair-Value Score by ZIP")
    feat_cols = features
    X_all = listings[feat_cols].fillna(listings[feat_cols].median())
    listings = listings.copy()
    listings["predicted_price"] = np.exp(model.predict(X_all))
    listings["deviation_pct"] = ((listings["price"] - listings["predicted_price"])
                                  / listings["predicted_price"] * 100).round(1)
    listings["fair_value"] = pd.cut(
        listings["deviation_pct"],
        bins=[-200, -8, 8, 200],
        labels=["Undervalued", "Fair Value", "Overvalued"],
    )

    zip_summary = (
        listings.groupby(["zip_code", "city"])
        .agg(median_price=("price", "median"),
             median_pred=("predicted_price", "median"),
             avg_deviation=("deviation_pct", "mean"),
             count=("price", "count"))
        .reset_index()
    )
    zip_summary["avg_deviation"] = zip_summary["avg_deviation"].round(1)
    zip_summary["label"] = pd.cut(zip_summary["avg_deviation"],
                                   bins=[-200, -8, 8, 200],
                                   labels=["Undervalued", "Fair Value", "Overvalued"])

    color_map = {"Undervalued": "green", "Fair Value": "steelblue", "Overvalued": "crimson"}
    fig3 = px.bar(
        zip_summary.sort_values("avg_deviation"),
        x="zip_code", y="avg_deviation", color="label",
        color_discrete_map=color_map,
        labels={"avg_deviation": "Avg Deviation from Fair Value (%)", "zip_code": "ZIP"},
    )
    fig3.add_hline(y=8, line_dash="dot", line_color="red", annotation_text="Overvalued threshold")
    fig3.add_hline(y=-8, line_dash="dot", line_color="green", annotation_text="Undervalued threshold")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

with st.expander("📊 ZIP fair-value detail table"):
    show = zip_summary[["zip_code", "city", "median_price", "median_pred", "avg_deviation", "label", "count"]]
    show = show.rename(columns={"median_price": "Median Price", "median_pred": "Predicted",
                                 "avg_deviation": "Deviation %", "label": "Rating", "count": "Listings"})
    st.dataframe(
        show.style.format({"Median Price": "${:,.0f}", "Predicted": "${:,.0f}", "Deviation %": "{:.1f}%"}),
        use_container_width=True,
    )
