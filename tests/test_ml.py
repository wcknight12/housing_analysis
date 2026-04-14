"""
Tests for the MLlib feature engineering pipeline.
"""

import os
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LISTINGS_PATH = os.path.join(DATA_DIR, "zillow_tx_listings.csv")


@pytest.fixture(scope="module")
def spark():
    from etl.ingest import get_spark
    sess = get_spark("TestML")
    yield sess
    sess.stop()


@pytest.fixture(scope="module")
def listings_df(spark):
    from etl.ingest import read_listings_csv
    from etl.transform import clean_listings
    raw = read_listings_csv(spark, LISTINGS_PATH)
    return clean_listings(raw)


class TestBuildFeaturePipeline:
    def test_pipeline_fits_and_transforms(self, listings_df):
        from ml.features import build_feature_pipeline, FEATURES_COL
        pipeline = build_feature_pipeline()
        model = pipeline.fit(listings_df)
        output = model.transform(listings_df)
        assert FEATURES_COL in output.columns

    def test_no_null_features(self, listings_df):
        from ml.features import build_feature_pipeline, FEATURES_COL
        from pyspark.sql import functions as F
        pipeline = build_feature_pipeline()
        model = pipeline.fit(listings_df)
        output = model.transform(listings_df)
        null_count = output.filter(F.col(FEATURES_COL).isNull()).count()
        assert null_count == 0

    def test_pipeline_without_scaling(self, listings_df):
        from ml.features import build_feature_pipeline, FEATURES_COL
        pipeline = build_feature_pipeline(scale=False)
        model = pipeline.fit(listings_df)
        output = model.transform(listings_df)
        assert FEATURES_COL in output.columns


class TestAddLogPrice:
    def test_adds_log_price_column(self, listings_df):
        from ml.features import add_log_price, LABEL_COL
        result = add_log_price(listings_df)
        assert f"log_{LABEL_COL}" in result.columns

    def test_log_price_positive(self, listings_df):
        from ml.features import add_log_price, LABEL_COL
        from pyspark.sql import functions as F
        result = add_log_price(listings_df)
        assert result.filter(F.col(f"log_{LABEL_COL}").isNull()).count() == 0


class TestTrainModel:
    def test_returns_three_items(self, listings_df):
        from ml.price_model import train_model
        result = train_model(listings_df, test_fraction=0.2)
        assert len(result) == 3

    def test_metrics_keys(self, listings_df):
        from ml.price_model import train_model
        _, _, metrics = train_model(listings_df, test_fraction=0.2)
        assert {"r2", "mae", "rmse"}.issubset(metrics.keys())

    def test_r2_positive(self, listings_df):
        from ml.price_model import train_model
        _, _, metrics = train_model(listings_df, test_fraction=0.2)
        assert metrics["r2"] > 0

    def test_prediction_column_in_output(self, listings_df):
        from ml.price_model import train_model
        _, test_df, _ = train_model(listings_df, test_fraction=0.2)
        assert "prediction" in test_df.columns


class TestPredict:
    def test_adds_predicted_price(self, listings_df):
        from ml.price_model import train_model, predict
        model, _, _ = train_model(listings_df, test_fraction=0.2)
        output = predict(model, listings_df, use_log_price=True)
        assert "predicted_price" in output.columns

    def test_predicted_prices_positive(self, listings_df):
        from ml.price_model import train_model, predict
        from pyspark.sql import functions as F
        model, _, _ = train_model(listings_df, test_fraction=0.2)
        output = predict(model, listings_df)
        assert output.filter(F.col("predicted_price") <= 0).count() == 0


class TestFairValue:
    def test_adds_deviation_column(self, listings_df):
        from ml.price_model import train_model
        from ml.fair_value import compute_fair_value_score
        model, _, _ = train_model(listings_df, test_fraction=0.2)
        scored = compute_fair_value_score(listings_df, model)
        assert "price_deviation" in scored.columns

    def test_adds_label_column(self, listings_df):
        from ml.price_model import train_model
        from ml.fair_value import compute_fair_value_score
        model, _, _ = train_model(listings_df, test_fraction=0.2)
        scored = compute_fair_value_score(listings_df, model)
        assert "fair_value_label" in scored.columns

    def test_valid_labels(self, listings_df):
        from ml.price_model import train_model
        from ml.fair_value import compute_fair_value_score
        model, _, _ = train_model(listings_df, test_fraction=0.2)
        scored = compute_fair_value_score(listings_df, model)
        labels = {r["fair_value_label"] for r in scored.select("fair_value_label").distinct().collect()}
        assert labels.issubset({"Undervalued", "Fair Value", "Overvalued"})
