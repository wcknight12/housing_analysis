"""
Spark Structured Streaming consumer for Texas Housing live data.

Reads from Kafka topics and processes:
  - Mortgage rate updates  → rolling average window
  - New listing alerts     → real-time price tier counts per city

Requires a running Kafka broker (see docker-compose.yml).

Run:
    python -m streaming.consumer --topic listings
"""

import argparse
import json
import logging
import os

from streaming.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    SPARK_STREAMING_CHECKPOINT_DIR,
    SPARK_STREAMING_TRIGGER_SECONDS,
    TOPIC_MORTGAGE_RATES,
    TOPIC_NEW_LISTINGS,
)

logger = logging.getLogger(__name__)


def _get_spark_with_kafka():
    """
    Create a SparkSession with the Kafka connector on the classpath.

    The spark-sql-kafka package is downloaded automatically via spark.jars.packages.
    """
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("TexasHousingStreaming")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config(
            "spark.streaming.stopGracefullyOnShutdown", "true"
        )
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Schema definitions for parsed JSON payloads
# ---------------------------------------------------------------------------

def _mortgage_rate_schema():
    from pyspark.sql.types import (
        DoubleType, StringType, StructField, StructType,
    )
    return StructType(
        [
            StructField("timestamp", StringType()),
            StructField("rate_30yr_fixed", DoubleType()),
            StructField("rate_15yr_fixed", DoubleType()),
            StructField("rate_5yr_arm", DoubleType()),
            StructField("source", StringType()),
        ]
    )


def _listing_schema():
    from pyspark.sql.types import (
        DoubleType, IntegerType, StringType, StructField, StructType,
    )
    return StructType(
        [
            StructField("timestamp", StringType()),
            StructField("listing_id", StringType()),
            StructField("city", StringType()),
            StructField("zip_code", StringType()),
            StructField("sqft", IntegerType()),
            StructField("bedrooms", IntegerType()),
            StructField("bathrooms", DoubleType()),
            StructField("price", IntegerType()),
            StructField("price_per_sqft", DoubleType()),
            StructField("year_built", IntegerType()),
            StructField("listing_type", StringType()),
        ]
    )


# ---------------------------------------------------------------------------
# Streaming queries
# ---------------------------------------------------------------------------

def stream_mortgage_rates(spark, output_mode: str = "console"):
    """
    Read mortgage-rate events from Kafka and compute a 5-minute rolling
    average for the 30-year fixed rate.

    Parameters
    ----------
    spark : SparkSession
    output_mode : str
        'console' (default) or 'memory' (for testing).
    """
    from pyspark.sql import functions as F

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC_MORTGAGE_RATES)
        .option("startingOffsets", "latest")
        .load()
    )

    schema = _mortgage_rate_schema()
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), schema).alias("data")
    ).select("data.*").withColumn(
        "event_time", F.to_timestamp(F.col("timestamp"))
    )

    windowed = (
        parsed.withWatermark("event_time", "2 minutes")
        .groupBy(F.window("event_time", "5 minutes", "1 minute"))
        .agg(
            F.round(F.avg("rate_30yr_fixed"), 3).alias("avg_30yr"),
            F.round(F.avg("rate_15yr_fixed"), 3).alias("avg_15yr"),
            F.round(F.avg("rate_5yr_arm"), 3).alias("avg_arm"),
            F.count("*").alias("event_count"),
        )
    )

    checkpoint = os.path.join(SPARK_STREAMING_CHECKPOINT_DIR, "mortgage_rates")

    if output_mode == "memory":
        query = (
            windowed.writeStream.outputMode("update")
            .format("memory")
            .queryName("mortgage_rates")
            .trigger(processingTime=f"{SPARK_STREAMING_TRIGGER_SECONDS} seconds")
            .option("checkpointLocation", checkpoint)
            .start()
        )
    else:
        query = (
            windowed.writeStream.outputMode("update")
            .format("console")
            .option("truncate", False)
            .trigger(processingTime=f"{SPARK_STREAMING_TRIGGER_SECONDS} seconds")
            .option("checkpointLocation", checkpoint)
            .start()
        )
    return query


def stream_new_listings(spark, output_mode: str = "console"):
    """
    Read new-listing events from Kafka and compute counts by city and
    price tier in a 1-minute tumbling window.

    Parameters
    ----------
    spark : SparkSession
    output_mode : str
        'console' or 'memory'.
    """
    from pyspark.sql import functions as F

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC_NEW_LISTINGS)
        .option("startingOffsets", "latest")
        .load()
    )

    schema = _listing_schema()
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), schema).alias("data")
    ).select("data.*").withColumn(
        "event_time", F.to_timestamp(F.col("timestamp"))
    )

    # Add price tier
    parsed = parsed.withColumn(
        "price_tier",
        F.when(F.col("price") < 250_000, "Budget")
        .when(F.col("price") < 450_000, "Mid-Range")
        .when(F.col("price") < 700_000, "Premium")
        .otherwise("Luxury"),
    )

    aggregated = (
        parsed.withWatermark("event_time", "2 minutes")
        .groupBy(
            F.window("event_time", "1 minute"),
            F.col("city"),
            F.col("price_tier"),
        )
        .agg(
            F.count("listing_id").alias("listing_count"),
            F.round(F.avg("price"), 0).alias("avg_price"),
            F.round(F.avg("sqft"), 0).alias("avg_sqft"),
        )
    )

    checkpoint = os.path.join(SPARK_STREAMING_CHECKPOINT_DIR, "new_listings")

    if output_mode == "memory":
        query = (
            aggregated.writeStream.outputMode("update")
            .format("memory")
            .queryName("new_listings")
            .trigger(processingTime=f"{SPARK_STREAMING_TRIGGER_SECONDS} seconds")
            .option("checkpointLocation", checkpoint)
            .start()
        )
    else:
        query = (
            aggregated.writeStream.outputMode("update")
            .format("console")
            .option("truncate", False)
            .trigger(processingTime=f"{SPARK_STREAMING_TRIGGER_SECONDS} seconds")
            .option("checkpointLocation", checkpoint)
            .start()
        )
    return query


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Texas Housing Structured Streaming")
    parser.add_argument(
        "--topic",
        choices=["rates", "listings", "all"],
        default="all",
    )
    args = parser.parse_args()

    spark = _get_spark_with_kafka()

    queries = []
    if args.topic in ("rates", "all"):
        queries.append(stream_mortgage_rates(spark))
    if args.topic in ("listings", "all"):
        queries.append(stream_new_listings(spark))

    for q in queries:
        q.awaitTermination()
