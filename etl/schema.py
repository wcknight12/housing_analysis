"""
Spark schema definitions for Zillow CSV and listing data.
"""

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    DateType,
)

# Metadata columns present in every Zillow ZHVI export
ZHVI_META_COLS = [
    "RegionID",
    "SizeRank",
    "RegionName",
    "RegionType",
    "StateName",
    "State",
    "City",
    "Metro",
    "CountyName",
]

ZHVI_META_SCHEMA = StructType(
    [
        StructField("RegionID", IntegerType(), True),
        StructField("SizeRank", IntegerType(), True),
        StructField("RegionName", StringType(), True),
        StructField("RegionType", StringType(), True),
        StructField("StateName", StringType(), True),
        StructField("State", StringType(), True),
        StructField("City", StringType(), True),
        StructField("Metro", StringType(), True),
        StructField("CountyName", StringType(), True),
    ]
)

# Schema for the pivoted / melted long-form ZHVI table
ZHVI_LONG_SCHEMA = StructType(
    [
        StructField("zip_code", StringType(), False),
        StructField("city", StringType(), True),
        StructField("metro", StringType(), True),
        StructField("state", StringType(), True),
        StructField("date", StringType(), True),
        StructField("zhvi", DoubleType(), True),
    ]
)

# Schema for individual listing records
LISTING_SCHEMA = StructType(
    [
        StructField("city", StringType(), True),
        StructField("zip_code", StringType(), True),
        StructField("metro", StringType(), True),
        StructField("sqft", IntegerType(), True),
        StructField("bedrooms", IntegerType(), True),
        StructField("bathrooms", DoubleType(), True),
        StructField("lot_size_acres", DoubleType(), True),
        StructField("year_built", IntegerType(), True),
        StructField("age_years", IntegerType(), True),
        StructField("school_rating", DoubleType(), True),
        StructField("distance_from_center_miles", DoubleType(), True),
        StructField("property_tax_rate", DoubleType(), True),
        StructField("price", IntegerType(), True),
        StructField("price_per_sqft", DoubleType(), True),
        StructField("listing_date", StringType(), True),
    ]
)

# Census ACS schema
CENSUS_SCHEMA = StructType(
    [
        StructField("zip_code", StringType(), False),
        StructField("city_name", StringType(), True),
        StructField("metro", StringType(), True),
        StructField("state", StringType(), True),
        StructField("population", IntegerType(), True),
        StructField("median_household_income", DoubleType(), True),
        StructField("median_age", DoubleType(), True),
        StructField("pct_bachelor_or_higher", DoubleType(), True),
        StructField("pct_below_poverty", DoubleType(), True),
        StructField("median_commute_minutes", DoubleType(), True),
        StructField("pct_owner_occupied", DoubleType(), True),
        StructField("unemployment_rate", DoubleType(), True),
        StructField("school_rating", DoubleType(), True),
        StructField("walk_score", IntegerType(), True),
        StructField("transit_score", IntegerType(), True),
        StructField("crime_index", DoubleType(), True),
        StructField("park_access_score", DoubleType(), True),
        StructField("air_quality_index", DoubleType(), True),
    ]
)
