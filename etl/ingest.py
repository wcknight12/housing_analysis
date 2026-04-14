"""
Zillow CSV ingestion for the Texas Housing ETL pipeline.

Reads Zillow ZHVI wide-format CSVs and listing CSVs into Spark DataFrames,
validates schema, and returns raw DataFrames ready for transformation.
"""

import os
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from etl.schema import ZHVI_META_COLS, LISTING_SCHEMA, CENSUS_SCHEMA


def get_spark(app_name: str = "TexasHousingETL") -> SparkSession:
    """
    Create or retrieve a SparkSession configured for local mode.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def read_zhvi_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Read a Zillow ZHVI wide-format CSV (one row per ZIP, date columns as values).

    Parameters
    ----------
    spark : SparkSession
    path : str
        Absolute or relative path to the Zillow ZHVI CSV file.

    Returns
    -------
    DataFrame
        Raw wide-format DataFrame with metadata columns and YYYY-MM-DD value columns.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If required metadata columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ZHVI CSV not found: {path}")

    df = spark.read.csv(path, header=True, inferSchema=True)

    missing = [c for c in ZHVI_META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ZHVI CSV missing required columns: {missing}")

    return df


def read_listings_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Read a Zillow-style individual listings CSV.

    Parameters
    ----------
    spark : SparkSession
    path : str
        Path to the listings CSV file.

    Returns
    -------
    DataFrame
        Listings DataFrame, cast to the LISTING_SCHEMA types where columns exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Listings CSV not found: {path}")

    df = spark.read.csv(path, header=True, inferSchema=True)

    # Cast columns to canonical types from schema (only those present)
    schema_map = {f.name: f.dataType for f in LISTING_SCHEMA.fields}
    for col_name, dtype in schema_map.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(dtype))

    return df


def read_census_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Read Census ACS data CSV.

    Parameters
    ----------
    spark : SparkSession
    path : str
        Path to the Census ACS CSV file.

    Returns
    -------
    DataFrame
        Census DataFrame cast to CENSUS_SCHEMA types.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Census CSV not found: {path}")

    df = spark.read.csv(path, header=True, inferSchema=True)

    schema_map = {f.name: f.dataType for f in CENSUS_SCHEMA.fields}
    for col_name, dtype in schema_map.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(dtype))

    return df


def validate_row_count(df: DataFrame, min_rows: int = 1, name: str = "DataFrame") -> DataFrame:
    """
    Assert a DataFrame has at least *min_rows* rows.

    Returns the unchanged DataFrame so it can be used in a pipeline chain.
    """
    count = df.count()
    if count < min_rows:
        raise ValueError(f"{name} has {count} rows; expected at least {min_rows}.")
    return df
