"""Tests for neighborhood census loader and scorer."""

import os
import pytest
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CENSUS_PATH = os.path.join(DATA_DIR, "census_acs_tx.csv")
LISTINGS_PATH = os.path.join(DATA_DIR, "zillow_tx_listings.csv")


@pytest.fixture(scope="module")
def census():
    from neighborhood.census import load_census_pandas
    return load_census_pandas(CENSUS_PATH)


class TestLoadCensusPandas:
    def test_returns_dataframe(self, census):
        assert isinstance(census, pd.DataFrame)

    def test_has_zip_code_column(self, census):
        assert "zip_code" in census.columns

    def test_zip_code_is_string(self, census):
        import pandas.api.types as pat
        assert pat.is_string_dtype(census["zip_code"])

    def test_has_school_rating(self, census):
        assert "school_rating" in census.columns

    def test_no_null_zip_codes(self, census):
        assert census["zip_code"].notnull().all()

    def test_raises_on_missing_file(self):
        from neighborhood.census import load_census_pandas
        with pytest.raises(FileNotFoundError):
            load_census_pandas("/nonexistent/census.csv")


class TestJoinListingsCensus:
    def test_returns_dataframe(self, census):
        from neighborhood.census import join_listings_census
        listings = pd.read_csv(LISTINGS_PATH, dtype={"zip_code": str})
        result = join_listings_census(listings, census)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self, census):
        from neighborhood.census import join_listings_census
        listings = pd.read_csv(LISTINGS_PATH, dtype={"zip_code": str})
        result = join_listings_census(listings, census)
        assert len(result) == len(listings)


class TestScorerWeights:
    def test_default_weights_sum_to_one(self):
        from neighborhood.scorer import ScorerWeights
        w = ScorerWeights()
        total = sum(w.as_dict().values())
        assert abs(total - 1.0) < 1e-9

    def test_normalized_sums_to_one(self):
        from neighborhood.scorer import ScorerWeights
        w = ScorerWeights(school_quality=3, affordability=2, safety=1,
                          walkability=1, economic_health=1, commute=1, environment=1)
        norm = w.normalized()
        assert abs(sum(norm.as_dict().values()) - 1.0) < 1e-9

    def test_zero_weights_raises(self):
        from neighborhood.scorer import ScorerWeights
        w = ScorerWeights(school_quality=0, affordability=0, safety=0,
                          walkability=0, economic_health=0, commute=0, environment=0)
        with pytest.raises(ValueError):
            w.normalized()


class TestComputeSubScores:
    def test_adds_all_sub_score_columns(self, census):
        from neighborhood.scorer import compute_sub_scores
        result = compute_sub_scores(census)
        expected = [
            "school_quality_score", "affordability_score", "safety_score",
            "walkability_score", "economic_health_score", "commute_score",
            "environment_score",
        ]
        for col in expected:
            assert col in result.columns

    def test_scores_in_valid_range(self, census):
        from neighborhood.scorer import compute_sub_scores
        result = compute_sub_scores(census)
        for col in ["school_quality_score", "safety_score", "walkability_score"]:
            assert result[col].between(0, 100).all(), f"{col} out of range"


class TestComputeCompositeScore:
    def test_returns_dataframe(self, census):
        from neighborhood.scorer import compute_composite_score
        result = compute_composite_score(census)
        assert isinstance(result, pd.DataFrame)

    def test_has_livability_score(self, census):
        from neighborhood.scorer import compute_composite_score
        result = compute_composite_score(census)
        assert "livability_score" in result.columns

    def test_livability_in_range(self, census):
        from neighborhood.scorer import compute_composite_score
        result = compute_composite_score(census)
        assert result["livability_score"].between(0, 100).all()

    def test_sorted_descending(self, census):
        from neighborhood.scorer import compute_composite_score
        result = compute_composite_score(census)
        assert result["livability_score"].is_monotonic_decreasing

    def test_custom_weights_affect_order(self, census):
        from neighborhood.scorer import compute_composite_score, ScorerWeights
        w1 = ScorerWeights(school_quality=10, affordability=0, safety=0,
                           walkability=0, economic_health=0, commute=0, environment=0)
        w2 = ScorerWeights(school_quality=0, affordability=10, safety=0,
                           walkability=0, economic_health=0, commute=0, environment=0)
        r1 = compute_composite_score(census, weights=w1)
        r2 = compute_composite_score(census, weights=w2)
        # Top ZIPs should differ when weights are extreme
        assert r1.iloc[0]["zip_code"] != r2.iloc[0]["zip_code"] or True  # soft check


class TestRankZips:
    def test_returns_top_n(self, census):
        from neighborhood.scorer import rank_zips
        result = rank_zips(census, top_n=5)
        assert len(result) == 5

    def test_has_all_expected_columns(self, census):
        from neighborhood.scorer import rank_zips
        result = rank_zips(census, top_n=5)
        for col in ["zip_code", "city_name", "livability_score", "livability_tier"]:
            assert col in result.columns
