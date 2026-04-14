"""
Kafka producers for live mortgage rate updates and new listing alerts.

Run this module directly to start the simulation producers:

    python -m streaming.producer --mode all

Environment variables:
    KAFKA_BOOTSTRAP_SERVERS  (default: localhost:9092)
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Optional

from streaming.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    MORTGAGE_RATE_INTERVAL_SECONDS,
    TOPIC_MORTGAGE_RATES,
    TOPIC_NEW_LISTINGS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mortgage rate simulation state
# ---------------------------------------------------------------------------

_CURRENT_30YR_RATE = 6.85   # seed value (%)
_CURRENT_15YR_RATE = 6.10

_TEXAS_CITIES = ["Austin", "Houston", "Dallas", "Fort Worth", "San Antonio"]
_TX_ZIPS = [
    "78701", "78704", "77007", "77019", "75201", "75205",
    "76107", "78209", "78258", "77380",
]


def _make_kafka_producer():
    """Create a KafkaProducer; raises ImportError / NoBrokersAvailable cleanly."""
    from kafka import KafkaProducer  # type: ignore
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


# ---------------------------------------------------------------------------
# Message factories
# ---------------------------------------------------------------------------

def _mortgage_rate_message() -> dict:
    """Simulate a small random walk on mortgage rates."""
    global _CURRENT_30YR_RATE, _CURRENT_15YR_RATE
    _CURRENT_30YR_RATE += random.gauss(0, 0.04)
    _CURRENT_30YR_RATE = round(max(3.0, min(10.0, _CURRENT_30YR_RATE)), 3)
    _CURRENT_15YR_RATE = round(_CURRENT_30YR_RATE - random.uniform(0.5, 0.9), 3)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rate_30yr_fixed": _CURRENT_30YR_RATE,
        "rate_15yr_fixed": _CURRENT_15YR_RATE,
        "rate_5yr_arm": round(_CURRENT_30YR_RATE - random.uniform(0.8, 1.4), 3),
        "source": "simulated",
    }


def _new_listing_message() -> dict:
    """Simulate a new listing event."""
    city = random.choice(_TEXAS_CITIES)
    sqft = random.randint(1_000, 4_500)
    beds = random.choice([2, 3, 3, 4, 4, 5])
    baths = random.choice([2.0, 2.5, 3.0, 3.5])
    price = int(sqft * random.uniform(140, 340))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "listing_id": f"TX-{random.randint(100000, 999999)}",
        "city": city,
        "zip_code": random.choice(_TX_ZIPS),
        "sqft": sqft,
        "bedrooms": beds,
        "bathrooms": baths,
        "price": price,
        "price_per_sqft": round(price / sqft, 2),
        "year_built": random.randint(1970, 2023),
        "listing_type": "new",
    }


# ---------------------------------------------------------------------------
# Producer loops
# ---------------------------------------------------------------------------

def run_mortgage_rate_producer(interval: int = MORTGAGE_RATE_INTERVAL_SECONDS) -> None:
    """
    Continuously publish simulated mortgage rate updates to Kafka.

    Runs until interrupted (Ctrl-C / SIGTERM).
    """
    producer = _make_kafka_producer()
    logger.info("Mortgage rate producer started → topic: %s", TOPIC_MORTGAGE_RATES)
    try:
        while True:
            msg = _mortgage_rate_message()
            producer.send(TOPIC_MORTGAGE_RATES, key="rates", value=msg)
            logger.debug("Sent: %s", msg)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Mortgage rate producer stopped.")
    finally:
        producer.flush()
        producer.close()


def run_listing_producer(interval: float = 2.0) -> None:
    """
    Continuously publish simulated new listing events to Kafka.

    Runs until interrupted.
    """
    producer = _make_kafka_producer()
    logger.info("Listing producer started → topic: %s", TOPIC_NEW_LISTINGS)
    try:
        while True:
            msg = _new_listing_message()
            producer.send(TOPIC_NEW_LISTINGS, key=msg["zip_code"], value=msg)
            logger.debug("Sent: %s", msg)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Listing producer stopped.")
    finally:
        producer.flush()
        producer.close()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Texas Housing Kafka Producers")
    parser.add_argument(
        "--mode",
        choices=["rates", "listings", "all"],
        default="all",
        help="Which producer(s) to run.",
    )
    args = parser.parse_args()

    threads = []
    if args.mode in ("rates", "all"):
        threads.append(threading.Thread(target=run_mortgage_rate_producer, daemon=True))
    if args.mode in ("listings", "all"):
        threads.append(threading.Thread(target=run_listing_producer, daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()
