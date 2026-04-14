"""
Kafka and Spark Structured Streaming configuration.
"""

import os

# ---------------------------------------------------------------------------
# Kafka broker settings (override via environment variables)
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_GROUP_ID: str = os.getenv("KAFKA_GROUP_ID", "texas-housing-consumers")

# ---------------------------------------------------------------------------
# Topic names
# ---------------------------------------------------------------------------

TOPIC_MORTGAGE_RATES = "mortgage-rate-updates"
TOPIC_NEW_LISTINGS = "new-listing-alerts"
TOPIC_PRICE_ALERTS = "price-alerts"

# ---------------------------------------------------------------------------
# Producer settings
# ---------------------------------------------------------------------------

PRODUCER_BATCH_SIZE = 16_384          # bytes
PRODUCER_LINGER_MS = 5               # ms
PRODUCER_COMPRESSION = "gzip"

# ---------------------------------------------------------------------------
# Streaming settings
# ---------------------------------------------------------------------------

SPARK_STREAMING_CHECKPOINT_DIR: str = os.getenv(
    "SPARK_STREAMING_CHECKPOINT_DIR", "/tmp/housing_streaming_checkpoint"
)
SPARK_STREAMING_TRIGGER_SECONDS: int = int(
    os.getenv("SPARK_STREAMING_TRIGGER_SECONDS", "30")
)

# Mortgage rate poll interval in seconds (simulated)
MORTGAGE_RATE_INTERVAL_SECONDS: int = int(
    os.getenv("MORTGAGE_RATE_INTERVAL_SECONDS", "60")
)
