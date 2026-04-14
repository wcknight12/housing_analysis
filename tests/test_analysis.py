"""Tests for the analysis module."""

import pytest
import pandas as pd
import numpy as np

from housing_analysis.data_loader import load_data
from housing_analysis.analysis import (
    descriptive_stats,
    price_distribution_by_city,
    correlation_with_price,
    run_linear_regression,
    run_random_forest,
    run_gradient_boosting,
    affordability_analysis,
    price_efficiency_analysis,
    combined_feature_importance,
    run_full_analysis,
    NUMERIC_FEATURES,
)


@pytest.fixture(scope="module")
def df():
    return load_data(n_samples=400, random_seed=0)


@pytest.fixture(scope="module")
def results(df):
    return run_full_analysis(df)


# -------------------------------------------------------------------------
# descriptive_stats
# -------------------------------------------------------------------------

class TestDescriptiveStats:
    def test_returns_dict_with_expected_keys(self, df):
        stats = descriptive_stats(df)
        assert "overall" in stats
        assert "by_city" in stats

    def test_overall_is_dataframe(self, df):
        stats = descriptive_stats(df)
        assert isinstance(stats["overall"], pd.DataFrame)

    def test_by_city_has_all_cities(self, df):
        stats = descriptive_stats(df)
        by_city = stats["by_city"]
        assert set(by_city.index) == set(df["city"].unique())

    def test_by_city_expected_columns(self, df):
        stats = descriptive_stats(df)
        required = {"median_price", "mean_price", "median_price_per_sqft",
                    "median_sqft", "avg_school_rating", "count"}
        assert required.issubset(set(stats["by_city"].columns))

    def test_median_prices_positive(self, df):
        stats = descriptive_stats(df)
        assert (stats["by_city"]["median_price"] > 0).all()


# -------------------------------------------------------------------------
# price_distribution_by_city
# -------------------------------------------------------------------------

class TestPriceDistributionByCity:
    def test_returns_dataframe(self, df):
        result = price_distribution_by_city(df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df):
        result = price_distribution_by_city(df)
        assert set(result.columns) == {"Q1", "median", "Q3"}

    def test_q1_le_median_le_q3(self, df):
        result = price_distribution_by_city(df)
        assert (result["Q1"] <= result["median"]).all()
        assert (result["median"] <= result["Q3"]).all()


# -------------------------------------------------------------------------
# correlation_with_price
# -------------------------------------------------------------------------

class TestCorrelationWithPrice:
    def test_returns_series(self, df):
        corr = correlation_with_price(df)
        assert isinstance(corr, pd.Series)

    def test_all_numeric_features_present(self, df):
        corr = correlation_with_price(df)
        for feat in NUMERIC_FEATURES:
            assert feat in corr.index

    def test_price_not_in_index(self, df):
        corr = correlation_with_price(df)
        assert "price" not in corr.index

    def test_values_in_valid_range(self, df):
        corr = correlation_with_price(df)
        assert corr.between(-1.0, 1.0).all()

    def test_sqft_has_positive_correlation(self, df):
        corr = correlation_with_price(df)
        assert corr["sqft"] > 0


# -------------------------------------------------------------------------
# run_linear_regression
# -------------------------------------------------------------------------

class TestLinearRegression:
    def test_returns_three_items(self, df):
        result = run_linear_regression(df)
        assert len(result) == 3

    def test_coef_dataframe_columns(self, df):
        _, coef_df, _ = run_linear_regression(df)
        assert set(coef_df.columns) == {"feature", "coefficient"}

    def test_all_features_in_coef(self, df):
        _, coef_df, _ = run_linear_regression(df)
        assert set(coef_df["feature"]) == set(NUMERIC_FEATURES)

    def test_metrics_keys(self, df):
        _, _, metrics = run_linear_regression(df)
        assert {"r2", "mae", "rmse", "cv_r2"}.issubset(metrics.keys())

    def test_r2_is_positive(self, df):
        _, _, metrics = run_linear_regression(df)
        assert metrics["r2"] > 0

    def test_mae_is_positive(self, df):
        _, _, metrics = run_linear_regression(df)
        assert metrics["mae"] > 0


# -------------------------------------------------------------------------
# run_random_forest
# -------------------------------------------------------------------------

class TestRandomForest:
    def test_returns_three_items(self, df):
        result = run_random_forest(df)
        assert len(result) == 3

    def test_importance_columns(self, df):
        _, imp_df, _ = run_random_forest(df)
        assert set(imp_df.columns) == {"feature", "importance"}

    def test_importances_sum_to_one(self, df):
        _, imp_df, _ = run_random_forest(df)
        assert abs(imp_df["importance"].sum() - 1.0) < 1e-6

    def test_r2_above_threshold(self, df):
        _, _, metrics = run_random_forest(df)
        assert metrics["r2"] > 0.5

    def test_metrics_keys(self, df):
        _, _, metrics = run_random_forest(df)
        assert {"r2", "mae", "rmse", "cv_r2"}.issubset(metrics.keys())


# -------------------------------------------------------------------------
# run_gradient_boosting
# -------------------------------------------------------------------------

class TestGradientBoosting:
    def test_returns_three_items(self, df):
        result = run_gradient_boosting(df)
        assert len(result) == 3

    def test_importance_columns(self, df):
        _, imp_df, _ = run_gradient_boosting(df)
        assert set(imp_df.columns) == {"feature", "importance"}

    def test_importances_sum_to_one(self, df):
        _, imp_df, _ = run_gradient_boosting(df)
        assert abs(imp_df["importance"].sum() - 1.0) < 1e-6

    def test_r2_above_threshold(self, df):
        _, _, metrics = run_gradient_boosting(df)
        assert metrics["r2"] > 0.4


# -------------------------------------------------------------------------
# affordability_analysis
# -------------------------------------------------------------------------

class TestAffordabilityAnalysis:
    def test_returns_dataframe(self, df):
        result = affordability_analysis(df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df):
        result = affordability_analysis(df)
        required = {"city", "median_price", "max_affordable_price", "affordable_pct"}
        assert required.issubset(set(result.columns))

    def test_affordable_pct_range(self, df):
        result = affordability_analysis(df)
        assert result["affordable_pct"].between(0, 100).all()

    def test_higher_income_increases_affordability(self, df):
        low_income = affordability_analysis(df, annual_income=60_000)
        high_income = affordability_analysis(df, annual_income=150_000)
        low_pct = low_income["affordable_pct"].mean()
        high_pct = high_income["affordable_pct"].mean()
        assert high_pct > low_pct

    def test_all_cities_present(self, df):
        result = affordability_analysis(df)
        assert set(result["city"]) == set(df["city"].unique())


# -------------------------------------------------------------------------
# price_efficiency_analysis
# -------------------------------------------------------------------------

class TestPriceEfficiencyAnalysis:
    def test_returns_dataframe(self, df):
        result = price_efficiency_analysis(df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df):
        result = price_efficiency_analysis(df)
        required = {"city", "sqft_tier", "median_price", "median_ppsf", "count"}
        assert required.issubset(set(result.columns))

    def test_no_negative_prices(self, df):
        result = price_efficiency_analysis(df)
        assert (result["median_price"] > 0).all()


# -------------------------------------------------------------------------
# combined_feature_importance
# -------------------------------------------------------------------------

class TestCombinedFeatureImportance:
    def test_returns_dataframe(self, df):
        _, rf_imp, _ = run_random_forest(df)
        _, gb_imp, _ = run_gradient_boosting(df)
        _, lr_coef, _ = run_linear_regression(df)
        result = combined_feature_importance(rf_imp, gb_imp, lr_coef)
        assert isinstance(result, pd.DataFrame)

    def test_all_features_present(self, df):
        _, rf_imp, _ = run_random_forest(df)
        _, gb_imp, _ = run_gradient_boosting(df)
        _, lr_coef, _ = run_linear_regression(df)
        result = combined_feature_importance(rf_imp, gb_imp, lr_coef)
        assert set(result["feature"]) == set(NUMERIC_FEATURES)

    def test_avg_rank_column_present(self, df):
        _, rf_imp, _ = run_random_forest(df)
        _, gb_imp, _ = run_gradient_boosting(df)
        _, lr_coef, _ = run_linear_regression(df)
        result = combined_feature_importance(rf_imp, gb_imp, lr_coef)
        assert "avg_rank" in result.columns


# -------------------------------------------------------------------------
# run_full_analysis
# -------------------------------------------------------------------------

class TestRunFullAnalysis:
    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_expected_keys(self, results):
        required = {
            "descriptive", "correlations", "linear_regression",
            "random_forest", "gradient_boosting", "combined_importance",
            "affordability", "price_efficiency",
        }
        assert required.issubset(results.keys())

    def test_combined_importance_is_dataframe(self, results):
        assert isinstance(results["combined_importance"], pd.DataFrame)

    def test_affordability_is_dataframe(self, results):
        assert isinstance(results["affordability"], pd.DataFrame)
