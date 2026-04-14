"""Tests for streaming config and producer message factories."""

import pytest


class TestStreamingConfig:
    def test_imports_cleanly(self):
        from streaming.config import (
            KAFKA_BOOTSTRAP_SERVERS,
            TOPIC_MORTGAGE_RATES,
            TOPIC_NEW_LISTINGS,
            TOPIC_PRICE_ALERTS,
        )
        assert isinstance(KAFKA_BOOTSTRAP_SERVERS, str)
        assert TOPIC_MORTGAGE_RATES == "mortgage-rate-updates"
        assert TOPIC_NEW_LISTINGS == "new-listing-alerts"
        assert TOPIC_PRICE_ALERTS == "price-alerts"


class TestMortgageRateMessage:
    def test_returns_dict(self):
        from streaming.producer import _mortgage_rate_message
        msg = _mortgage_rate_message()
        assert isinstance(msg, dict)

    def test_has_required_keys(self):
        from streaming.producer import _mortgage_rate_message
        msg = _mortgage_rate_message()
        assert {"timestamp", "rate_30yr_fixed", "rate_15yr_fixed"}.issubset(msg.keys())

    def test_rates_in_valid_range(self):
        from streaming.producer import _mortgage_rate_message
        for _ in range(20):
            msg = _mortgage_rate_message()
            assert 3.0 <= msg["rate_30yr_fixed"] <= 10.0
            assert 3.0 <= msg["rate_15yr_fixed"] <= 10.0


class TestNewListingMessage:
    def test_returns_dict(self):
        from streaming.producer import _new_listing_message
        msg = _new_listing_message()
        assert isinstance(msg, dict)

    def test_has_required_keys(self):
        from streaming.producer import _new_listing_message
        msg = _new_listing_message()
        required = {"timestamp", "listing_id", "city", "zip_code",
                    "sqft", "bedrooms", "price", "listing_type"}
        assert required.issubset(msg.keys())

    def test_price_positive(self):
        from streaming.producer import _new_listing_message
        for _ in range(10):
            msg = _new_listing_message()
            assert msg["price"] > 0

    def test_listing_id_has_tx_prefix(self):
        from streaming.producer import _new_listing_message
        for _ in range(10):
            msg = _new_listing_message()
            assert msg["listing_id"].startswith("TX-")

    def test_city_is_valid_texas_city(self):
        from streaming.producer import _new_listing_message, _TEXAS_CITIES
        for _ in range(20):
            msg = _new_listing_message()
            assert msg["city"] in _TEXAS_CITIES
