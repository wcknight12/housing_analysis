"""Tests for the insights module."""

import pytest
import pandas as pd

from housing_analysis.data_loader import load_data
from housing_analysis.analysis import run_full_analysis
from housing_analysis.insights import (
    top_price_drivers,
    best_value_cities,
    city_market_summary,
    recommend_for_buyer,
    generate_insights,
    run_insights,
)
from housing_analysis.analysis import NUMERIC_FEATURES


@pytest.fixture(scope="module")
def df():
    return load_data(n_samples=400, random_seed=1)


@pytest.fixture(scope="module")
def results(df):
    return run_full_analysis(df)


@pytest.fixture(scope="module")
def insight_results(results):
    return run_insights(results)


# -------------------------------------------------------------------------
# top_price_drivers
# -------------------------------------------------------------------------

class TestTopPriceDrivers:
    def test_returns_list(self, results):
        drivers = top_price_drivers(results["combined_importance"])
        assert isinstance(drivers, list)

    def test_default_length(self, results):
        drivers = top_price_drivers(results["combined_importance"])
        assert len(drivers) == 5

    def test_custom_length(self, results):
        for n in (1, 3, 8):
            drivers = top_price_drivers(results["combined_importance"], top_n=n)
            assert len(drivers) == n

    def test_drivers_are_valid_features(self, results):
        drivers = top_price_drivers(results["combined_importance"])
        for d in drivers:
            assert d in NUMERIC_FEATURES


# -------------------------------------------------------------------------
# best_value_cities
# -------------------------------------------------------------------------

class TestBestValueCities:
    def test_returns_list(self, results):
        cities = best_value_cities(results["affordability"])
        assert isinstance(cities, list)

    def test_all_cities_present(self, df, results):
        cities = best_value_cities(results["affordability"])
        assert set(cities) == set(df["city"].unique())

    def test_most_affordable_is_first(self, results):
        cities = best_value_cities(results["affordability"])
        affordability = results["affordability"].sort_values("affordable_pct", ascending=False)
        assert cities[0] == affordability.iloc[0]["city"]


# -------------------------------------------------------------------------
# city_market_summary
# -------------------------------------------------------------------------

class TestCityMarketSummary:
    def test_returns_dataframe(self, results):
        summary = city_market_summary(results["descriptive"]["by_city"])
        assert isinstance(summary, pd.DataFrame)

    def test_market_tier_column(self, results):
        summary = city_market_summary(results["descriptive"]["by_city"])
        assert "market_tier" in summary.columns

    def test_valid_market_tiers(self, results):
        summary = city_market_summary(results["descriptive"]["by_city"])
        valid_tiers = {"Premium", "Mid-Range", "Affordable"}
        assert set(summary["market_tier"].unique()).issubset(valid_tiers)


# -------------------------------------------------------------------------
# recommend_for_buyer
# -------------------------------------------------------------------------

class TestRecommendForBuyer:
    def test_returns_dataframe(self, df):
        result = recommend_for_buyer(
            budget=400_000,
            preferred_sqft=1_800,
            min_bedrooms=3,
            min_school_rating=6.0,
            df=df,
        )
        assert isinstance(result, pd.DataFrame)

    def test_results_within_budget(self, df):
        budget = 350_000
        result = recommend_for_buyer(
            budget=budget,
            preferred_sqft=1_800,
            min_bedrooms=3,
            min_school_rating=5.0,
            df=df,
        )
        if not result.empty:
            assert (result["price"] <= budget).all()

    def test_results_meet_bedroom_requirement(self, df):
        min_beds = 4
        result = recommend_for_buyer(
            budget=500_000,
            preferred_sqft=2_000,
            min_bedrooms=min_beds,
            min_school_rating=1.0,
            df=df,
        )
        if not result.empty:
            assert (result["bedrooms"] >= min_beds).all()

    def test_results_meet_school_requirement(self, df):
        min_school = 7.0
        result = recommend_for_buyer(
            budget=600_000,
            preferred_sqft=2_000,
            min_bedrooms=2,
            min_school_rating=min_school,
            df=df,
        )
        if not result.empty:
            assert (result["school_rating"] >= min_school).all()

    def test_max_20_results(self, df):
        result = recommend_for_buyer(
            budget=1_000_000,
            preferred_sqft=2_000,
            min_bedrooms=2,
            min_school_rating=1.0,
            df=df,
        )
        assert len(result) <= 20

    def test_sorted_by_value_score_descending(self, df):
        result = recommend_for_buyer(
            budget=500_000,
            preferred_sqft=1_800,
            min_bedrooms=3,
            min_school_rating=5.0,
            df=df,
        )
        if len(result) > 1:
            assert (result["value_score"].diff().dropna() <= 0).all()

    def test_empty_result_for_impossible_criteria(self, df):
        result = recommend_for_buyer(
            budget=1,
            preferred_sqft=1_800,
            min_bedrooms=3,
            min_school_rating=5.0,
            df=df,
        )
        assert result.empty

    def test_value_score_column_present(self, df):
        result = recommend_for_buyer(
            budget=500_000,
            preferred_sqft=1_800,
            min_bedrooms=3,
            min_school_rating=5.0,
            df=df,
        )
        if not result.empty:
            assert "value_score" in result.columns


# -------------------------------------------------------------------------
# generate_insights
# -------------------------------------------------------------------------

class TestGenerateInsights:
    def test_returns_list(self, results):
        insights = generate_insights(results)
        assert isinstance(insights, list)

    def test_non_empty(self, results):
        insights = generate_insights(results)
        assert len(insights) > 0

    def test_all_are_strings(self, results):
        insights = generate_insights(results)
        assert all(isinstance(i, str) for i in insights)

    def test_contains_city_mention(self, results):
        insights = generate_insights(results)
        combined = " ".join(insights)
        assert any(city in combined for city in ["Austin", "Houston", "Dallas", "San Antonio", "Fort Worth"])


# -------------------------------------------------------------------------
# run_insights
# -------------------------------------------------------------------------

class TestRunInsights:
    def test_returns_dict(self, insight_results):
        assert isinstance(insight_results, dict)

    def test_expected_keys(self, insight_results):
        required = {"insights", "city_summary", "best_value_cities", "top_drivers"}
        assert required.issubset(insight_results.keys())

    def test_insights_is_list(self, insight_results):
        assert isinstance(insight_results["insights"], list)

    def test_city_summary_is_dataframe(self, insight_results):
        assert isinstance(insight_results["city_summary"], pd.DataFrame)

    def test_best_value_cities_is_list(self, insight_results):
        assert isinstance(insight_results["best_value_cities"], list)

    def test_top_drivers_is_list(self, insight_results):
        assert isinstance(insight_results["top_drivers"], list)
