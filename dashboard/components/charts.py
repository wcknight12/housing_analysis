"""Reusable Plotly chart components for the Streamlit dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


CITY_COLORS = {
    "Austin": "#1f77b4",
    "Dallas": "#ff7f0e",
    "Fort Worth": "#2ca02c",
    "Houston": "#d62728",
    "San Antonio": "#9467bd",
}


def price_trend_line(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    """Line chart for price trends."""
    fig = px.line(
        df, x=x, y=y, color=color,
        title=title,
        color_discrete_map=CITY_COLORS,
        labels={y: "Median Home Value ($)", x: "Date"},
    )
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="City / Metro",
        yaxis_tickformat="$,.0f",
    )
    return fig


def price_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    """Sorted bar chart."""
    fig = px.bar(
        df.sort_values(y, ascending=False),
        x=x, y=y, color=color,
        title=title,
        color_discrete_map=CITY_COLORS,
        labels={y: "Median Price ($)", x: ""},
    )
    fig.update_layout(yaxis_tickformat="$,.0f", showlegend=False)
    return fig


def scatter_price_sqft(df: pd.DataFrame, color_col: str = "city"):
    """Scatter: price vs sqft coloured by city."""
    fig = px.scatter(
        df, x="sqft", y="price",
        color=color_col,
        opacity=0.45,
        title="Home Price vs. Square Footage",
        labels={"sqft": "Square Footage", "price": "Sale Price ($)"},
        color_discrete_map=CITY_COLORS,
        hover_data=["bedrooms", "bathrooms", "zip_code"],
    )
    fig.update_layout(yaxis_tickformat="$,.0f")
    return fig


def feature_importance_bar(importances: dict, title: str = "Feature Importance"):
    """Horizontal bar of feature importances."""
    labels = {
        "sqft": "Square Footage", "school_rating": "School Rating",
        "distance_from_center_miles": "Distance from Center",
        "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
        "age_years": "Age of Home", "lot_size_acres": "Lot Size",
        "property_tax_rate": "Property Tax Rate",
    }
    df = pd.DataFrame(
        [{"feature": labels.get(k, k), "importance": v} for k, v in importances.items()]
    ).sort_values("importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h", title=title)
    fig.update_layout(yaxis_title="", xaxis_title="Importance Score")
    return fig


def neighborhood_radar(scores: dict, zip_code: str):
    """Radar chart of sub-scores for a single ZIP."""
    categories = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])  # close the loop
    categories.append(categories[0])
    fig = go.Figure(
        go.Scatterpolar(
            r=values, theta=categories, fill="toself",
            line_color="#1f77b4",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Livability Radar — ZIP {zip_code}",
        showlegend=False,
    )
    return fig


def affordability_gauge(affordable_pct: float, city: str):
    """Gauge chart showing % of homes affordable."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=affordable_pct,
            title={"text": f"{city} Affordability (% homes within budget)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "steelblue"},
                "steps": [
                    {"range": [0, 30], "color": "#ffcccc"},
                    {"range": [30, 60], "color": "#fff3cc"},
                    {"range": [60, 100], "color": "#ccffcc"},
                ],
            },
            number={"suffix": "%"},
        )
    )
    return fig


def buy_vs_wait_chart(df: pd.DataFrame):
    """Line chart comparing total cost of buying now vs. waiting N years."""
    fig = px.line(
        df, x="wait_years", y=["buy_now_total_cost", "wait_total_cost"],
        title="Buy Now vs. Wait — Total 10-Year Cost Comparison",
        labels={"wait_years": "Years to Wait Before Buying", "value": "Total Cost ($)"},
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
    )
    fig.update_layout(yaxis_tickformat="$,.0f", legend_title_text="Scenario")
    fig.for_each_trace(lambda t: t.update(
        name="Buy Now" if "buy_now" in t.name else "Wait & Buy Later"
    ))
    return fig
