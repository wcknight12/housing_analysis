"""
Tests for the ETL ingest and transform modules.

Uses a shared SparkSession fixture (scope=session) to avoid repeated JVM starts.
"""

import os
import pytest
import pandas as pd

# Shared Spark session for the whole test module
@pytest.fixture(scope="session")
def spark():
    from etl.ingest import get_spark
    sess = get_spark("TestETL")
    yield sess
    sess.stop()


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZHVI_PATH = os.path.join(DATA_DIR, "zillow_tx_zhvi.csv")
LISTINGS_PATH = os.path.join(DATA_DIR, "zillow_tx_listings.csv")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestReadZhviCsv:
    def test_reads_successfully(self, spark):
        from etl.ingest import read_zhvi_csv
        df = read_zhvi_csv(spark, ZHVI_PATH)
        assert df.count() > 0

    def test_has_meta_columns(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.schema import ZHVI_META_COLS
        df = read_zhvi_csv(spark, ZHVI_PATH)
        for col in ZHVI_META_COLS:
            assert col in df.columns

    def test_raises_on_missing_file(self, spark):
        from etl.ingest import read_zhvi_csv
        with pytest.raises(FileNotFoundError):
            read_zhvi_csv(spark, "/nonexistent/path.csv")


class TestReadListingsCsv:
    def test_reads_successfully(self, spark):
        from etl.ingest import read_listings_csv
        df = read_listings_csv(spark, LISTINGS_PATH)
        assert df.count() > 0

    def test_has_price_column(self, spark):
        from etl.ingest import read_listings_csv
        df = read_listings_csv(spark, LISTINGS_PATH)
        assert "price" in df.columns


class TestValidateRowCount:
    def test_passes_on_sufficient_rows(self, spark):
        from etl.ingest import read_zhvi_csv, validate_row_count
        df = read_zhvi_csv(spark, ZHVI_PATH)
        result = validate_row_count(df, min_rows=1)
        assert result is df

    def test_raises_on_insufficient_rows(self, spark):
        from etl.ingest import validate_row_count
        empty = spark.createDataFrame([], schema="id INT")
        with pytest.raises(ValueError):
            validate_row_count(empty, min_rows=1)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

class TestPivotZhviWideLong:
    def test_produces_rows(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.transform import pivot_zhvi_wide_to_long
        wide = read_zhvi_csv(spark, ZHVI_PATH)
        long_df = pivot_zhvi_wide_to_long(wide)
        assert long_df.count() > 0

    def test_expected_columns(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.transform import pivot_zhvi_wide_to_long
        wide = read_zhvi_csv(spark, ZHVI_PATH)
        long_df = pivot_zhvi_wide_to_long(wide)
        for col in ["zip_code", "city", "metro", "date", "zhvi"]:
            assert col in long_df.columns

    def test_no_null_zhvi(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.transform import pivot_zhvi_wide_to_long
        from pyspark.sql import functions as F
        wide = read_zhvi_csv(spark, ZHVI_PATH)
        long_df = pivot_zhvi_wide_to_long(wide)
        null_count = long_df.filter(F.col("zhvi").isNull()).count()
        assert null_count == 0


class TestMonthlyMedianByMetro:
    def test_returns_rows(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.transform import pivot_zhvi_wide_to_long, monthly_median_by_metro
        wide = read_zhvi_csv(spark, ZHVI_PATH)
        long_df = pivot_zhvi_wide_to_long(wide)
        result = monthly_median_by_metro(long_df)
        assert result.count() > 0

    def test_has_expected_columns(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.transform import pivot_zhvi_wide_to_long, monthly_median_by_metro
        wide = read_zhvi_csv(spark, ZHVI_PATH)
        long_df = pivot_zhvi_wide_to_long(wide)
        result = monthly_median_by_metro(long_df)
        for col in ["metro", "year", "month", "median_zhvi"]:
            assert col in result.columns


class TestYoyPriceChange:
    def test_adds_yoy_column(self, spark):
        from etl.ingest import read_zhvi_csv
        from etl.transform import pivot_zhvi_wide_to_long, yoy_price_change
        wide = read_zhvi_csv(spark, ZHVI_PATH)
        long_df = pivot_zhvi_wide_to_long(wide)
        result = yoy_price_change(long_df)
        assert "yoy_pct_change" in result.columns


class TestCleanListings:
    def test_filters_low_prices(self, spark):
        from etl.ingest import read_listings_csv
        from etl.transform import clean_listings
        from pyspark.sql import functions as F
        raw = read_listings_csv(spark, LISTINGS_PATH)
        clean = clean_listings(raw)
        assert clean.filter(F.col("price") < 50_000).count() == 0

    def test_adds_price_tier(self, spark):
        from etl.ingest import read_listings_csv
        from etl.transform import clean_listings
        raw = read_listings_csv(spark, LISTINGS_PATH)
        clean = clean_listings(raw)
        assert "price_tier" in clean.columns

    def test_valid_price_tiers(self, spark):
        from etl.ingest import read_listings_csv
        from etl.transform import clean_listings
        raw = read_listings_csv(spark, LISTINGS_PATH)
        clean = clean_listings(raw)
        tiers = {r["price_tier"] for r in clean.select("price_tier").distinct().collect()}
        assert tiers.issubset({"Budget", "Mid-Range", "Premium", "Luxury"})


class TestZipListingSummary:
    def test_returns_rows(self, spark):
        from etl.ingest import read_listings_csv
        from etl.transform import clean_listings, zip_listing_summary
        raw = read_listings_csv(spark, LISTINGS_PATH)
        clean = clean_listings(raw)
        summary = zip_listing_summary(clean)
        assert summary.count() > 0

    def test_expected_columns(self, spark):
        from etl.ingest import read_listings_csv
        from etl.transform import clean_listings, zip_listing_summary
        raw = read_listings_csv(spark, LISTINGS_PATH)
        clean = clean_listings(raw)
        summary = zip_listing_summary(clean)
        for col in ["zip_code", "city", "median_price", "listing_count"]:
            assert col in summary.columns
