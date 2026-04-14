"""
Census ACS data loader.

Provides helpers to load the Census ACS CSV into pandas (for the dashboard)
and PySpark (for the ETL pipeline) and merge it with listing data.
"""

import os
from typing import Optional

import pandas as pd


DEFAULT_CENSUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "census_acs_tx.csv"
)


def load_census_pandas(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load Census ACS data into a pandas DataFrame.

    Parameters
    ----------
    path : str, optional
        Path to census_acs_tx.csv. Defaults to data/census_acs_tx.csv
        relative to this package.

    Returns
    -------
    pd.DataFrame
        Census ACS records with zip_code as string.
    """
    if path is None:
        path = DEFAULT_CENSUS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Census CSV not found: {path}")
    df = pd.read_csv(path, dtype={"zip_code": str})
    return df


def load_census_spark(spark, path: Optional[str] = None):
    """
    Load Census ACS data into a PySpark DataFrame.

    Parameters
    ----------
    spark : SparkSession
    path : str, optional

    Returns
    -------
    pyspark.sql.DataFrame
    """
    if path is None:
        path = DEFAULT_CENSUS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Census CSV not found: {path}")
    from etl.schema import CENSUS_SCHEMA
    from pyspark.sql import functions as F
    df = spark.read.csv(path, header=True, schema=CENSUS_SCHEMA)
    return df


def join_listings_census(
    listings_df: pd.DataFrame,
    census_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join listings with Census ACS data on zip_code.

    Parameters
    ----------
    listings_df : pd.DataFrame
        Listings data with a 'zip_code' column.
    census_df : pd.DataFrame
        Census ACS data with a 'zip_code' column.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame; listings without a matching ZIP keep NaN for ACS cols.
    """
    listings_df = listings_df.copy()
    listings_df["zip_code"] = listings_df["zip_code"].astype(str)
    census_df = census_df.copy()
    census_df["zip_code"] = census_df["zip_code"].astype(str)

    # Only include census columns that are NOT already in listings (except zip_code for the join key)
    census_cols = ["zip_code"] + [
        c for c in census_df.columns
        if c != "zip_code" and c not in listings_df.columns
    ]
    return listings_df.merge(census_df[census_cols], on="zip_code", how="left")
