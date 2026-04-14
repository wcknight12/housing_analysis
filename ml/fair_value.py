"""
Fair-value scoring per ZIP code.

Compares each ZIP's median listing price against the model-predicted
fair value to produce an over/under-valued signal for consumers.
"""

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# Score labels
UNDERVALUED_THRESHOLD = -0.08   # > 8% below predicted = undervalued
OVERVALUED_THRESHOLD = 0.08     # > 8% above predicted = overvalued


def compute_fair_value_score(
    listings_df: DataFrame,
    model: PipelineModel,
    use_log_price: bool = True,
) -> DataFrame:
    """
    For each listing, compute how far the actual price deviates from
    the model's predicted fair value.

    Adds columns:
      - predicted_price : model fair-value estimate
      - price_deviation : (price - predicted) / predicted
      - fair_value_label: 'Undervalued' | 'Fair Value' | 'Overvalued'

    Parameters
    ----------
    listings_df : DataFrame
        Cleaned listings DataFrame.
    model : PipelineModel
        Fitted price prediction pipeline.
    use_log_price : bool
        Must match the value used when training the model.

    Returns
    -------
    DataFrame
        Input DataFrame with fair-value columns added.
    """
    from ml.price_model import predict

    scored = predict(model, listings_df, use_log_price=use_log_price)

    scored = scored.withColumn(
        "price_deviation",
        F.round(
            (F.col("price").cast("double") - F.col("predicted_price"))
            / F.col("predicted_price"),
            4,
        ),
    )

    scored = scored.withColumn(
        "fair_value_label",
        F.when(F.col("price_deviation") < UNDERVALUED_THRESHOLD, "Undervalued")
        .when(F.col("price_deviation") > OVERVALUED_THRESHOLD, "Overvalued")
        .otherwise("Fair Value"),
    )

    return scored


def zip_fair_value_summary(scored_df: DataFrame) -> DataFrame:
    """
    Aggregate fair-value scores to ZIP level.

    Returns one row per ZIP with:
      zip_code, city, metro, median_price, median_predicted,
      median_deviation_pct, fair_value_label, listing_count
    """
    summary = (
        scored_df.groupBy("zip_code", "city", "metro")
        .agg(
            F.percentile_approx("price", 0.5).alias("median_price"),
            F.percentile_approx("predicted_price", 0.5).alias("median_predicted"),
            F.round(F.avg("price_deviation") * 100, 2).alias("avg_deviation_pct"),
            F.count("price").alias("listing_count"),
        )
    )

    summary = summary.withColumn(
        "fair_value_label",
        F.when(F.col("avg_deviation_pct") < UNDERVALUED_THRESHOLD * 100, "Undervalued")
        .when(F.col("avg_deviation_pct") > OVERVALUED_THRESHOLD * 100, "Overvalued")
        .otherwise("Fair Value"),
    )

    return summary.orderBy("zip_code")


def metro_fair_value_summary(scored_df: DataFrame) -> DataFrame:
    """
    Aggregate fair-value scores to metro level.
    """
    return (
        scored_df.groupBy("metro")
        .agg(
            F.percentile_approx("price", 0.5).alias("median_price"),
            F.percentile_approx("predicted_price", 0.5).alias("median_predicted"),
            F.round(F.avg("price_deviation") * 100, 2).alias("avg_deviation_pct"),
            F.count("price").alias("listing_count"),
        )
        .orderBy("metro")
    )
