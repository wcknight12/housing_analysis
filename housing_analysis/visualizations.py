"""
Visualization module for Texas housing market analysis.

Creates publication-quality charts that communicate price drivers and
consumer insights for single-family homes across Texas metros.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

PALETTE = "Set2"
FIGURE_DPI = 120
TEXAS_CITIES = ["Austin", "Dallas", "Fort Worth", "Houston", "San Antonio"]

_CITY_COLORS = dict(zip(TEXAS_CITIES, sns.color_palette(PALETTE, len(TEXAS_CITIES))))


def _money(x, _pos=None):
    """Formatter: convert raw dollar value to $XK or $XM."""
    if x >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"


def _setup_style():
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Box-plot of home-price distributions by Texas city.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    order = (
        df.groupby("city")["price"].median().sort_values(ascending=False).index.tolist()
    )
    palette = [_CITY_COLORS.get(c, "steelblue") for c in order]

    sns.boxplot(
        data=df,
        x="city",
        y="price",
        order=order,
        hue="city",
        palette=palette,
        ax=ax,
        legend=False,
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.set_title("Home Price Distribution by Texas City", fontweight="bold")
    ax.set_xlabel("City")
    ax.set_ylabel("Sale Price")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_price_vs_sqft(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Scatter plot: price vs. square footage, coloured by city.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for city, group in df.groupby("city"):
        ax.scatter(
            group["sqft"],
            group["price"],
            label=city,
            alpha=0.35,
            s=18,
            color=_CITY_COLORS.get(city, "steelblue"),
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Home Price vs. Square Footage by City", fontweight="bold")
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Sale Price")
    ax.legend(title="City", loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_feature_importance(
    combined_importance: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Horizontal bar chart of average feature importance rank across all models.
    Lower avg_rank = more important.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    feature_labels = {
        "sqft": "Square Footage",
        "school_rating": "School Rating",
        "distance_from_center_miles": "Distance from Center",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "age_years": "Age of Home",
        "lot_size_acres": "Lot Size",
        "property_tax_rate": "Property Tax Rate",
    }

    data = combined_importance.copy()
    data["label"] = data["feature"].map(feature_labels)
    # Invert rank so higher bar = more important
    data["importance_score"] = (data["avg_rank"].max() + 1) - data["avg_rank"]
    data = data.sort_values("importance_score", ascending=True)

    bars = ax.barh(
        data["label"],
        data["importance_score"],
        color=sns.color_palette(PALETTE, len(data)),
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_title("Price Driver Importance (All Models Combined)", fontweight="bold")
    ax.set_xlabel("Relative Importance Score (higher = more important)")
    ax.set_xlim(0, data["importance_score"].max() * 1.15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap of Pearson correlations among numeric features including price.
    """
    _setup_style()
    from housing_analysis.analysis import NUMERIC_FEATURES

    cols = NUMERIC_FEATURES + ["price"]
    corr_matrix = df[cols].corr().round(2)

    rename_map = {
        "sqft": "Sq Ft",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "lot_size_acres": "Lot",
        "age_years": "Age",
        "school_rating": "School",
        "distance_from_center_miles": "Distance",
        "property_tax_rate": "Tax Rate",
        "price": "Price",
    }
    corr_matrix = corr_matrix.rename(columns=rename_map, index=rename_map)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_affordability(affordability: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Grouped bar chart: median price vs. max affordable price by city.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    cities = affordability["city"].tolist()
    x = np.arange(len(cities))
    width = 0.35

    bars1 = ax.bar(x - width / 2, affordability["median_price"], width,
                   label="Median Home Price", color=sns.color_palette(PALETTE)[0])
    bars2 = ax.bar(x + width / 2, affordability["max_affordable_price"], width,
                   label="Max Affordable ($85k HHI, 28% rule)", color=sns.color_palette(PALETTE)[1])

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_title("Median Home Price vs. Affordability Threshold by City", fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_price_per_sqft_by_city(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Violin plot of price-per-sqft distributions by city.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    order = (
        df.groupby("city")["price_per_sqft"].median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    palette = [_CITY_COLORS.get(c, "steelblue") for c in order]

    sns.violinplot(
        data=df,
        x="city",
        y="price_per_sqft",
        order=order,
        hue="city",
        palette=palette,
        ax=ax,
        legend=False,
        cut=0,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.set_title("Price per Square Foot Distribution by City", fontweight="bold")
    ax.set_xlabel("City")
    ax.set_ylabel("Price / Sq Ft")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_school_vs_price(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Scatter plot: school rating vs. price, with regression line.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    for city, group in df.groupby("city"):
        ax.scatter(
            group["school_rating"],
            group["price"],
            label=city,
            alpha=0.30,
            s=15,
            color=_CITY_COLORS.get(city, "steelblue"),
        )

    # Overall regression line
    x = df["school_rating"].values
    y = df["price"].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, "k--", linewidth=2, label="Trend line")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.set_title("School District Rating vs. Home Price", fontweight="bold")
    ax.set_xlabel("School Rating (1–10)")
    ax.set_ylabel("Sale Price")
    ax.legend(title="City", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_age_vs_price(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Scatter plot: home age vs. price with regression line.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        df["age_years"], df["price"], alpha=0.25, s=15,
        color=sns.color_palette(PALETTE)[2]
    )

    x = df["age_years"].values
    y = df["price"].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=2, label="Trend line")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_money))
    ax.set_title("Home Age vs. Sale Price", fontweight="bold")
    ax.set_xlabel("Age of Home (years)")
    ax.set_ylabel("Sale Price")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Generate all charts
# ---------------------------------------------------------------------------

def generate_all_charts(
    df: pd.DataFrame,
    results: Dict,
    output_dir: str = "charts",
) -> Dict[str, str]:
    """
    Generate and save all charts to *output_dir*.

    Returns
    -------
    dict mapping chart name to saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    chart_fns = [
        ("price_distribution", plot_price_distribution, (df,)),
        ("price_vs_sqft", plot_price_vs_sqft, (df,)),
        ("feature_importance", plot_feature_importance, (results["combined_importance"],)),
        ("correlation_heatmap", plot_correlation_heatmap, (df,)),
        ("affordability", plot_affordability, (results["affordability"],)),
        ("price_per_sqft", plot_price_per_sqft_by_city, (df,)),
        ("school_vs_price", plot_school_vs_price, (df,)),
        ("age_vs_price", plot_age_vs_price, (df,)),
    ]

    for name, fn, args in chart_fns:
        path = os.path.join(output_dir, f"{name}.png")
        fig = fn(*args, save_path=path)
        plt.close(fig)
        paths[name] = path

    return paths
