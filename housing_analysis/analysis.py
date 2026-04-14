"""
Analysis module for Texas housing market price drivers.

Provides statistical analysis and machine-learning-based feature importance
to identify the key drivers of single-family home prices across Texas metros.
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Numeric feature columns used in regression / ML models
NUMERIC_FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "lot_size_acres",
    "age_years",
    "school_rating",
    "distance_from_center_miles",
    "property_tax_rate",
]


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute descriptive statistics for the full dataset and by city.

    Returns
    -------
    dict with keys:
      'overall'  – summary statistics for all numeric columns
      'by_city'  – median price, price_per_sqft, sqft, and school_rating per city
    """
    overall = df.describe()

    by_city = (
        df.groupby("city")
        .agg(
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_sqft=("sqft", "median"),
            avg_school_rating=("school_rating", "mean"),
            avg_distance=("distance_from_center_miles", "mean"),
            count=("price", "count"),
        )
        .sort_values("median_price", ascending=False)
        .round(2)
    )

    return {"overall": overall, "by_city": by_city}


def price_distribution_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Return price quartiles (25 %, 50 %, 75 %) per city."""
    return (
        df.groupby("city")["price"]
        .quantile([0.25, 0.50, 0.75])
        .unstack(level=1)
        .rename(columns={0.25: "Q1", 0.50: "median", 0.75: "Q3"})
        .sort_values("median", ascending=False)
        .round(0)
    )


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def correlation_with_price(df: pd.DataFrame) -> pd.Series:
    """
    Pearson correlation of all numeric features with *price*.

    Returns
    -------
    pd.Series sorted by absolute correlation (descending).
    """
    numeric_df = df[NUMERIC_FEATURES + ["price"]].copy()
    corr = numeric_df.corr()["price"].drop("price")
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------

def run_linear_regression(
    df: pd.DataFrame,
) -> Tuple[Ridge, pd.DataFrame, Dict[str, float]]:
    """
    Fit a Ridge regression model to predict log(price).

    Returns
    -------
    model : fitted Ridge model
    coef_df : DataFrame of feature names and standardised coefficients
    metrics : dict with 'r2', 'mae', 'rmse', 'cv_r2'
    """
    X = df[NUMERIC_FEATURES].copy()
    y = np.log(df["price"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

    coef_df = pd.DataFrame(
        {"feature": NUMERIC_FEATURES, "coefficient": model.coef_}
    ).sort_values("coefficient", key=abs, ascending=False)

    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 4),
        "cv_r2": round(cv_scores.mean(), 4),
    }

    return model, coef_df, metrics


# ---------------------------------------------------------------------------
# Tree-based feature importance
# ---------------------------------------------------------------------------

def run_random_forest(
    df: pd.DataFrame,
) -> Tuple[RandomForestRegressor, pd.DataFrame, Dict[str, float]]:
    """
    Fit a Random Forest to predict price and extract feature importances.

    Returns
    -------
    model : fitted RandomForestRegressor
    importance_df : DataFrame ranked by feature importance
    metrics : dict with 'r2', 'mae', 'rmse', 'cv_r2'
    """
    X = df[NUMERIC_FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    importance_df = pd.DataFrame(
        {"feature": NUMERIC_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 4),
        "cv_r2": round(cv_scores.mean(), 4),
    }

    return model, importance_df, metrics


def run_gradient_boosting(
    df: pd.DataFrame,
) -> Tuple[GradientBoostingRegressor, pd.DataFrame, Dict[str, float]]:
    """
    Fit a Gradient Boosting model to predict price and extract feature importances.

    Returns
    -------
    model : fitted GradientBoostingRegressor
    importance_df : DataFrame ranked by feature importance
    metrics : dict with 'r2', 'mae', 'rmse', 'cv_r2'
    """
    X = df[NUMERIC_FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    importance_df = pd.DataFrame(
        {"feature": NUMERIC_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 4),
        "cv_r2": round(cv_scores.mean(), 4),
    }

    return model, importance_df, metrics


# ---------------------------------------------------------------------------
# Affordability analysis
# ---------------------------------------------------------------------------

def affordability_analysis(df: pd.DataFrame, annual_income: float = 85_000) -> pd.DataFrame:
    """
    Estimate affordability by city using the 28 % gross-income rule.

    Parameters
    ----------
    df : pd.DataFrame
    annual_income : float
        Household annual income (default: Texas median ~$85 k).

    Returns
    -------
    pd.DataFrame with city, median_price, max_affordable_price, affordable_pct.
    """
    # 28 % rule: max monthly mortgage ≤ 28 % of gross monthly income
    monthly_income = annual_income / 12
    max_monthly_payment = monthly_income * 0.28

    # Assume 30-year fixed, 7 % interest, 20 % down
    rate_monthly = 0.07 / 12
    n_payments = 360
    max_loan = max_monthly_payment * (
        (1 - (1 + rate_monthly) ** -n_payments) / rate_monthly
    )
    max_affordable = max_loan / 0.80  # account for 20 % down payment

    by_city = df.groupby("city").agg(
        median_price=("price", "median"),
        count=("price", "count"),
    )

    by_city["max_affordable_price"] = round(max_affordable, 0)
    by_city["affordable_pct"] = (
        df.groupby("city")["price"].apply(lambda x: (x <= max_affordable).mean()) * 100
    ).round(1)

    return by_city.sort_values("affordable_pct", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Price-per-sqft efficiency
# ---------------------------------------------------------------------------

def price_efficiency_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify cities / segments offering the best price-per-sqft value.

    Returns a DataFrame with city, sqft tier, median price, and
    median price_per_sqft for each combination.
    """
    df = df.copy()
    df["sqft_tier"] = pd.cut(
        df["sqft"],
        bins=[0, 1_200, 1_800, 2_500, 3_500, np.inf],
        labels=["<1 200", "1 200–1 800", "1 800–2 500", "2 500–3 500", "3 500+"],
    )

    efficiency = (
        df.groupby(["city", "sqft_tier"], observed=True)
        .agg(
            median_price=("price", "median"),
            median_ppsf=("price_per_sqft", "median"),
            count=("price", "count"),
        )
        .reset_index()
        .sort_values(["city", "sqft_tier"])
        .round(2)
    )
    return efficiency


# ---------------------------------------------------------------------------
# Feature-importance summary across models
# ---------------------------------------------------------------------------

def combined_feature_importance(
    rf_importance: pd.DataFrame,
    gb_importance: pd.DataFrame,
    lr_coef: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine feature importances from RF, GB, and LR (absolute coefficients)
    into a single ranked summary.

    Returns
    -------
    pd.DataFrame with columns: feature, rf_importance, gb_importance,
    lr_abs_coef, avg_rank.
    """
    rf = rf_importance.set_index("feature")["importance"].rename("rf_importance")
    gb = gb_importance.set_index("feature")["importance"].rename("gb_importance")
    lr = lr_coef.set_index("feature")["coefficient"].abs().rename("lr_abs_coef")

    combined = pd.concat([rf, gb, lr], axis=1)

    # Rank within each model (1 = most important)
    for col in ["rf_importance", "gb_importance", "lr_abs_coef"]:
        combined[f"{col}_rank"] = combined[col].rank(ascending=False)

    combined["avg_rank"] = combined[
        ["rf_importance_rank", "gb_importance_rank", "lr_abs_coef_rank"]
    ].mean(axis=1)

    return (
        combined[["rf_importance", "gb_importance", "lr_abs_coef", "avg_rank"]]
        .sort_values("avg_rank")
        .reset_index()
        .round(4)
    )


# ---------------------------------------------------------------------------
# Run full analysis pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(df: pd.DataFrame) -> Dict:
    """
    Run all analyses and return results as a dictionary.

    Keys: 'descriptive', 'correlations', 'linear_regression',
          'random_forest', 'gradient_boosting', 'combined_importance',
          'affordability', 'price_efficiency'.
    """
    stats = descriptive_stats(df)
    corr = correlation_with_price(df)
    lr_model, lr_coef, lr_metrics = run_linear_regression(df)
    rf_model, rf_imp, rf_metrics = run_random_forest(df)
    gb_model, gb_imp, gb_metrics = run_gradient_boosting(df)
    combined_imp = combined_feature_importance(rf_imp, gb_imp, lr_coef)
    affordability = affordability_analysis(df)
    efficiency = price_efficiency_analysis(df)

    return {
        "descriptive": stats,
        "correlations": corr,
        "linear_regression": {"model": lr_model, "coef": lr_coef, "metrics": lr_metrics},
        "random_forest": {"model": rf_model, "importance": rf_imp, "metrics": rf_metrics},
        "gradient_boosting": {"model": gb_model, "importance": gb_imp, "metrics": gb_metrics},
        "combined_importance": combined_imp,
        "affordability": affordability,
        "price_efficiency": efficiency,
    }
