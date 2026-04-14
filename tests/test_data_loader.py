"""Tests for the data_loader module."""

import pytest
import pandas as pd
import numpy as np

from housing_analysis.data_loader import (
    load_data,
    generate_texas_housing_data,
    TEXAS_METROS,
)


class TestGenerateTexasHousingData:
    """Tests for generate_texas_housing_data()."""

    def test_returns_dataframe(self):
        df = generate_texas_housing_data(n_samples=50)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        for n in (10, 100, 500):
            df = generate_texas_housing_data(n_samples=n)
            assert len(df) == n

    def test_expected_columns(self):
        required = {
            "city", "sqft", "bedrooms", "bathrooms", "lot_size_acres",
            "year_built", "age_years", "school_rating",
            "distance_from_center_miles", "property_tax_rate",
            "price", "price_per_sqft",
        }
        df = generate_texas_housing_data(n_samples=50)
        assert required.issubset(set(df.columns))

    def test_cities_are_valid(self):
        df = generate_texas_housing_data(n_samples=200)
        assert set(df["city"].unique()).issubset(set(TEXAS_METROS.keys()))

    def test_all_cities_represented(self):
        """With enough samples, all five cities should appear."""
        df = generate_texas_housing_data(n_samples=500)
        assert set(df["city"].unique()) == set(TEXAS_METROS.keys())

    def test_positive_prices(self):
        df = generate_texas_housing_data(n_samples=200)
        assert (df["price"] > 0).all()

    def test_sqft_range(self):
        df = generate_texas_housing_data(n_samples=200)
        assert df["sqft"].between(800, 6_000).all()

    def test_bedrooms_range(self):
        df = generate_texas_housing_data(n_samples=200)
        assert df["bedrooms"].isin([2, 3, 4, 5]).all()

    def test_school_rating_range(self):
        df = generate_texas_housing_data(n_samples=200)
        assert df["school_rating"].between(1.0, 10.0).all()

    def test_reproducibility(self):
        df1 = generate_texas_housing_data(n_samples=100, random_seed=7)
        df2 = generate_texas_housing_data(n_samples=100, random_seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_texas_housing_data(n_samples=100, random_seed=1)
        df2 = generate_texas_housing_data(n_samples=100, random_seed=2)
        assert not df1["price"].equals(df2["price"])

    def test_age_years_consistent_with_year_built(self):
        df = generate_texas_housing_data(n_samples=100)
        current_year = 2024
        assert (df["age_years"] == current_year - df["year_built"]).all()

    def test_price_per_sqft_consistent(self):
        df = generate_texas_housing_data(n_samples=100)
        expected = (df["price"] / df["sqft"]).round(2)
        pd.testing.assert_series_equal(df["price_per_sqft"], expected, check_names=False)

    def test_no_null_values(self):
        df = generate_texas_housing_data(n_samples=100)
        assert df.isnull().sum().sum() == 0

    def test_property_tax_rate_range(self):
        df = generate_texas_housing_data(n_samples=200)
        assert df["property_tax_rate"].between(0.015, 0.030).all()

    def test_lot_size_positive(self):
        df = generate_texas_housing_data(n_samples=200)
        assert (df["lot_size_acres"] > 0).all()


class TestLoadData:
    """Tests for the public load_data() entry-point."""

    def test_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_default_sample_size(self):
        df = load_data()
        assert len(df) == 1_000

    def test_custom_sample_size(self):
        df = load_data(n_samples=250)
        assert len(df) == 250

    def test_seed_passes_through(self):
        df1 = load_data(n_samples=50, random_seed=99)
        df2 = load_data(n_samples=50, random_seed=99)
        pd.testing.assert_frame_equal(df1, df2)
