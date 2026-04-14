"""Tests for the ETL schema definitions."""

import pytest
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType

from etl.schema import (
    ZHVI_META_COLS,
    ZHVI_META_SCHEMA,
    ZHVI_LONG_SCHEMA,
    LISTING_SCHEMA,
    CENSUS_SCHEMA,
)


class TestZhviMetaCols:
    def test_is_list(self):
        assert isinstance(ZHVI_META_COLS, list)

    def test_contains_required_fields(self):
        required = {"RegionName", "City", "Metro", "State", "RegionID"}
        assert required.issubset(set(ZHVI_META_COLS))


class TestSchemas:
    def test_zhvi_meta_schema_is_struct(self):
        assert isinstance(ZHVI_META_SCHEMA, StructType)

    def test_zhvi_long_schema_has_zip_code(self):
        names = {f.name for f in ZHVI_LONG_SCHEMA.fields}
        assert "zip_code" in names
        assert "zhvi" in names

    def test_listing_schema_has_price(self):
        names = {f.name for f in LISTING_SCHEMA.fields}
        assert "price" in names
        assert "sqft" in names

    def test_census_schema_has_income(self):
        names = {f.name for f in CENSUS_SCHEMA.fields}
        assert "median_household_income" in names
        assert "school_rating" in names
