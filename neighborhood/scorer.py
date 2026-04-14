"""
Neighborhood livability composite scorer.

Computes a weighted livability score (0–100) for each ZIP code by combining
Census ACS metrics. Weights are user-adjustable (dashboard sliders).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "school_quality": 0.25,
    "affordability": 0.20,
    "safety": 0.15,
    "walkability": 0.10,
    "economic_health": 0.15,
    "commute": 0.10,
    "environment": 0.05,
}

WEIGHT_LABELS: Dict[str, str] = {
    "school_quality": "School Quality",
    "affordability": "Affordability",
    "safety": "Safety",
    "walkability": "Walkability & Transit",
    "economic_health": "Economic Health",
    "commute": "Commute Friendliness",
    "environment": "Environment & Parks",
}


@dataclass
class ScorerWeights:
    """User-adjustable weights for the livability composite scorer."""

    school_quality: float = 0.25
    affordability: float = 0.20
    safety: float = 0.15
    walkability: float = 0.10
    economic_health: float = 0.15
    commute: float = 0.10
    environment: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "school_quality": self.school_quality,
            "affordability": self.affordability,
            "safety": self.safety,
            "walkability": self.walkability,
            "economic_health": self.economic_health,
            "commute": self.commute,
            "environment": self.environment,
        }

    def normalized(self) -> "ScorerWeights":
        """Return a copy with weights re-normalised to sum to 1.0."""
        d = self.as_dict()
        total = sum(d.values())
        if total == 0:
            raise ValueError("All weights are zero.")
        return ScorerWeights(**{k: v / total for k, v in d.items()})


def _minmax_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    """Scale a series to [0, 100]. Optionally invert (lower raw = higher score)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    scaled = (series - mn) / (mx - mn) * 100
    return (100 - scaled) if invert else scaled


def compute_sub_scores(census_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute individual sub-scores (0–100) for each livability dimension.

    Input must have columns from the Census ACS dataset.

    Returns the input DataFrame with additional *_score columns:
      school_quality_score, affordability_score, safety_score,
      walkability_score, economic_health_score, commute_score, environment_score
    """
    df = census_df.copy()

    # School quality: school_rating (higher = better)
    df["school_quality_score"] = _minmax_scale(df["school_rating"])

    # Affordability: median_household_income / (pct_owner_occupied + 1e-3)
    # Lower poverty & higher income → better
    income_score = _minmax_scale(df["median_household_income"])
    poverty_score = _minmax_scale(df["pct_below_poverty"], invert=True)
    df["affordability_score"] = income_score * 0.6 + poverty_score * 0.4

    # Safety: crime_index (lower = safer)
    df["safety_score"] = _minmax_scale(df["crime_index"], invert=True)

    # Walkability: walk_score + transit_score (higher = better)
    walk = _minmax_scale(df["walk_score"])
    transit = _minmax_scale(df["transit_score"])
    df["walkability_score"] = walk * 0.6 + transit * 0.4

    # Economic health: employment + education
    employment_score = _minmax_scale(df["unemployment_rate"], invert=True)
    education_score = _minmax_scale(df["pct_bachelor_or_higher"])
    df["economic_health_score"] = employment_score * 0.5 + education_score * 0.5

    # Commute friendliness: shorter commute = higher score
    df["commute_score"] = _minmax_scale(df["median_commute_minutes"], invert=True)

    # Environment: park access (higher = better) + air quality (lower AQI = better)
    park = _minmax_scale(df["park_access_score"])
    air = _minmax_scale(df["air_quality_index"], invert=True)
    df["environment_score"] = park * 0.6 + air * 0.4

    return df


def compute_composite_score(
    census_df: pd.DataFrame,
    weights: Optional[ScorerWeights] = None,
) -> pd.DataFrame:
    """
    Compute a weighted livability composite score (0–100) for each ZIP.

    Parameters
    ----------
    census_df : pd.DataFrame
        Census ACS DataFrame (output of census.load_census_pandas()).
    weights : ScorerWeights, optional
        Dimension weights. Uses DEFAULT_WEIGHTS if not provided.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with sub-scores and a 'livability_score' column.
        Sorted by livability_score descending.
    """
    if weights is None:
        weights = ScorerWeights()

    w = weights.normalized().as_dict()
    df = compute_sub_scores(census_df)

    score_cols = {
        "school_quality": "school_quality_score",
        "affordability": "affordability_score",
        "safety": "safety_score",
        "walkability": "walkability_score",
        "economic_health": "economic_health_score",
        "commute": "commute_score",
        "environment": "environment_score",
    }

    df["livability_score"] = sum(
        w[dim] * df[col] for dim, col in score_cols.items()
    ).round(1)

    # Tier labels
    df["livability_tier"] = pd.cut(
        df["livability_score"],
        bins=[0, 40, 60, 75, 100],
        labels=["Developing", "Average", "Good", "Excellent"],
    ).astype(str)

    return df.sort_values("livability_score", ascending=False).reset_index(drop=True)


def rank_zips(
    census_df: pd.DataFrame,
    weights: Optional[ScorerWeights] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-N ZIPs ranked by livability score.

    Parameters
    ----------
    census_df : pd.DataFrame
    weights : ScorerWeights, optional
    top_n : int

    Returns
    -------
    pd.DataFrame
        Top-N ZIPs with livability_score and sub-scores.
    """
    scored = compute_composite_score(census_df, weights)
    cols = [
        "zip_code", "city_name", "metro", "livability_score", "livability_tier",
        "school_quality_score", "affordability_score", "safety_score",
        "walkability_score", "economic_health_score", "commute_score",
        "environment_score",
    ]
    return scored[cols].head(top_n).reset_index(drop=True)
