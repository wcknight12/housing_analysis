"""
PySpark transformations for the Texas Housing ETL pipeline.

Provides:
- ZHVI wide → long pivot
- Price trend aggregations (monthly, quarterly, YoY)
- Listing feature enrichment
- ZIP-level summary tables
"""

from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from etl.schema import ZHVI_META_COLS


# ---------------------------------------------------------------------------
# ZHVI: wide → long
# ---------------------------------------------------------------------------

def pivot_zhvi_wide_to_long(df: DataFrame) -> DataFrame:
    """
    Convert Zillow ZHVI wide-format DataFrame to long (tidy) format.

    Input columns:  RegionName, City, Metro, State, ..., 2019-01-31, 2019-02-28, ...
    Output columns: zip_code, city, metro, state, date (str YYYY-MM-DD), zhvi (double)

    Parameters
    ----------
    df : DataFrame
        Wide-format ZHVI DataFrame from ingest.read_zhvi_csv().

    Returns
    -------
    DataFrame
        Long-format DataFrame with one row per (ZIP, date).
    """
    date_cols = [c for c in df.columns if c not in ZHVI_META_COLS and c[0].isdigit()]

    if not date_cols:
        raise ValueError("No date columns found in ZHVI DataFrame.")

    # Rename metadata columns to snake_case
    df = (
        df.withColumnRenamed("RegionName", "zip_code")
        .withColumnRenamed("City", "city")
        .withColumnRenamed("Metro", "metro")
        .withColumnRenamed("State", "state")
    )

    # Stack date columns into (date, zhvi) pairs
    stack_expr = f"stack({len(date_cols)}, " + ", ".join(
        f"'{c}', `{c}`" for c in date_cols
    ) + ") as (date, zhvi)"

    long_df = df.select("zip_code", "city", "metro", "state", F.expr(stack_expr))

    long_df = (
        long_df.filter(F.col("zhvi").isNotNull())
        .withColumn("date", F.col("date").cast("string"))
        .withColumn("zhvi", F.col("zhvi").cast("double"))
    )
    return long_df


# ---------------------------------------------------------------------------
# Trend aggregations
# ---------------------------------------------------------------------------

def add_date_parts(df: DataFrame, date_col: str = "date") -> DataFrame:
    """Add year, month, quarter columns derived from a string date column."""
    return (
        df.withColumn("year", F.year(F.to_date(F.col(date_col))))
        .withColumn("month", F.month(F.to_date(F.col(date_col))))
        .withColumn("quarter", F.quarter(F.to_date(F.col(date_col))))
    )


def monthly_median_by_metro(df: DataFrame) -> DataFrame:
    """
    Compute median ZHVI per metro per month.

    Input: long ZHVI DataFrame (zip_code, metro, date, zhvi)
    Output: metro, year, month, median_zhvi — sorted ascending
    """
    df = add_date_parts(df)
    return (
        df.groupBy("metro", "year", "month")
        .agg(
            F.percentile_approx("zhvi", 0.5).alias("median_zhvi"),
            F.avg("zhvi").alias("mean_zhvi"),
            F.count("zip_code").alias("zip_count"),
        )
        .orderBy("metro", "year", "month")
    )


def quarterly_median_by_city(df: DataFrame) -> DataFrame:
    """Compute median ZHVI per city per quarter."""
    df = add_date_parts(df)
    return (
        df.groupBy("city", "year", "quarter")
        .agg(
            F.percentile_approx("zhvi", 0.5).alias("median_zhvi"),
            F.count("zip_code").alias("zip_count"),
        )
        .orderBy("city", "year", "quarter")
    )


def yoy_price_change(df: DataFrame) -> DataFrame:
    """
    Compute year-over-year % price change per ZIP using a 12-month lag window.

    Adds columns: zhvi_lag_12m, yoy_pct_change
    """
    df = df.withColumn("date_parsed", F.to_date(F.col("date")))

    window = (
        Window.partitionBy("zip_code")
        .orderBy(F.unix_date(F.col("date_parsed")))
    )

    df = df.withColumn(
        "zhvi_lag_12m",
        F.lag("zhvi", 12).over(window),
    )

    df = df.withColumn(
        "yoy_pct_change",
        F.when(
            F.col("zhvi_lag_12m").isNotNull() & (F.col("zhvi_lag_12m") > 0),
            F.round(
                (F.col("zhvi") - F.col("zhvi_lag_12m")) / F.col("zhvi_lag_12m") * 100,
                2,
            ),
        ).otherwise(None),
    )
    return df.drop("date_parsed")


# ---------------------------------------------------------------------------
# Listing enrichment
# ---------------------------------------------------------------------------

def clean_listings(df: DataFrame) -> DataFrame:
    """
    Clean and standardise a listings DataFrame.

    - Drop rows with null price or sqft
    - Cap price at $5M and sqft at 8,000
    - Add price_tier column (budget/mid/premium/luxury)
    """
    df = df.filter(F.col("price").isNotNull() & F.col("sqft").isNotNull())
    df = df.filter((F.col("price") > 50_000) & (F.col("price") < 5_000_000))
    df = df.filter((F.col("sqft") >= 400) & (F.col("sqft") <= 8_000))

    df = df.withColumn(
        "price_tier",
        F.when(F.col("price") < 250_000, "Budget")
        .when(F.col("price") < 450_000, "Mid-Range")
        .when(F.col("price") < 700_000, "Premium")
        .otherwise("Luxury"),
    )
    return df


def zip_listing_summary(listings_df: DataFrame) -> DataFrame:
    """
    Aggregate listing-level data to ZIP-level summary statistics.

    Returns: zip_code, city, metro, median_price, median_sqft,
             median_ppsf, avg_school_rating, listing_count
    """
    return (
        listings_df.groupBy("zip_code", "city", "metro")
        .agg(
            F.percentile_approx("price", 0.5).alias("median_price"),
            F.percentile_approx("sqft", 0.5).alias("median_sqft"),
            F.percentile_approx("price_per_sqft", 0.5).alias("median_ppsf"),
            F.avg("school_rating").alias("avg_school_rating"),
            F.count("price").alias("listing_count"),
        )
        .orderBy("zip_code")
    )


# ---------------------------------------------------------------------------
# Full ETL pipeline
# ---------------------------------------------------------------------------

def run_etl(
    zhvi_path: str,
    listings_path: str,
    spark: Optional[SparkSession] = None,
):
    """
    Run the full ETL pipeline.

    Parameters
    ----------
    zhvi_path : str
        Path to Zillow ZHVI wide-format CSV.
    listings_path : str
        Path to listings CSV.
    spark : SparkSession, optional
        Existing SparkSession; created if not provided.

    Returns
    -------
    dict with keys:
        'zhvi_long'    – long-format ZHVI DataFrame
        'monthly_metro'– monthly median ZHVI by metro
        'quarterly_city'– quarterly median ZHVI by city
        'yoy_zip'      – year-over-year change per ZIP
        'listings'     – cleaned listings DataFrame
        'zip_summary'  – ZIP-level listing summary
    """
    from etl.ingest import get_spark, read_zhvi_csv, read_listings_csv

    if spark is None:
        spark = get_spark()

    # Ingest
    zhvi_wide = read_zhvi_csv(spark, zhvi_path)
    listings_raw = read_listings_csv(spark, listings_path)

    # Transform
    zhvi_long = pivot_zhvi_wide_to_long(zhvi_wide)
    monthly = monthly_median_by_metro(zhvi_long)
    quarterly = quarterly_median_by_city(zhvi_long)
    yoy = yoy_price_change(zhvi_long)
    listings_clean = clean_listings(listings_raw)
    zip_summary = zip_listing_summary(listings_clean)

    return {
        "zhvi_long": zhvi_long,
        "monthly_metro": monthly,
        "quarterly_city": quarterly,
        "yoy_zip": yoy,
        "listings": listings_clean,
        "zip_summary": zip_summary,
    }
