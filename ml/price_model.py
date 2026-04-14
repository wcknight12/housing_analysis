"""
MLlib price prediction model for Texas single-family homes.

Uses a Gradient Boosted Tree (GBT) regressor trained on listing features.
Provides train/evaluate/predict helpers and model persistence.
"""

import os
from typing import Dict, Optional, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ml.features import (
    FEATURES_COL,
    LABEL_COL,
    NUMERIC_FEATURES,
    add_log_price,
    build_feature_pipeline,
)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_gbt_pipeline(
    max_iter: int = 50,
    max_depth: int = 5,
    step_size: float = 0.1,
    use_log_price: bool = True,
) -> Pipeline:
    """
    Combine the feature pipeline with a GBT regressor into a single Pipeline.

    Parameters
    ----------
    max_iter : int
        Number of boosting iterations.
    max_depth : int
        Maximum depth of each tree.
    step_size : float
        Learning rate.
    use_log_price : bool
        If True the model trains on log(price); predictions are exponentiated
        back to dollar values in predict().

    Returns
    -------
    pyspark.ml.Pipeline
        Unfitted end-to-end pipeline.
    """
    label = f"log_{LABEL_COL}" if use_log_price else LABEL_COL

    feature_pipeline = build_feature_pipeline(scale=False)

    gbt = GBTRegressor(
        featuresCol=FEATURES_COL,
        labelCol=label,
        maxIter=max_iter,
        maxDepth=max_depth,
        stepSize=step_size,
        seed=42,
    )

    # Re-assemble: feature_pipeline stages + GBT
    all_stages = feature_pipeline.getStages() + [gbt]
    return Pipeline(stages=all_stages)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_model(
    listings_df: DataFrame,
    test_fraction: float = 0.2,
    use_log_price: bool = True,
) -> Tuple[PipelineModel, DataFrame, Dict[str, float]]:
    """
    Train a GBT price prediction model on a listings DataFrame.

    Parameters
    ----------
    listings_df : DataFrame
        Cleaned listings with price, sqft, bedrooms, … columns.
    test_fraction : float
        Fraction of data reserved for evaluation.
    use_log_price : bool
        Train on log(price) for better numeric stability.

    Returns
    -------
    model : PipelineModel
        Fitted pipeline model.
    test_df : DataFrame
        Hold-out test set with 'prediction' column.
    metrics : dict
        Keys: rmse, mae, r2 (on original dollar scale).
    """
    label_col = f"log_{LABEL_COL}" if use_log_price else LABEL_COL

    if use_log_price:
        listings_df = add_log_price(listings_df)

    train_df, test_df = listings_df.randomSplit(
        [1 - test_fraction, test_fraction], seed=42
    )

    pipeline = build_gbt_pipeline(use_log_price=use_log_price)
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)

    if use_log_price:
        predictions = predictions.withColumn(
            "prediction_price", F.exp(F.col("prediction"))
        )
        eval_col = "prediction_price"
        true_col = LABEL_COL
    else:
        eval_col = "prediction"
        true_col = LABEL_COL

    evaluator_rmse = RegressionEvaluator(
        labelCol=true_col, predictionCol=eval_col, metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol=true_col, predictionCol=eval_col, metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=true_col, predictionCol=eval_col, metricName="r2"
    )

    metrics = {
        "rmse": round(evaluator_rmse.evaluate(predictions), 2),
        "mae": round(evaluator_mae.evaluate(predictions), 2),
        "r2": round(evaluator_r2.evaluate(predictions), 4),
    }

    return model, predictions, metrics


def predict(
    model: PipelineModel,
    df: DataFrame,
    use_log_price: bool = True,
) -> DataFrame:
    """
    Run inference on a DataFrame with listing features.

    Returns the DataFrame with a 'predicted_price' column added.
    """
    if use_log_price:
        df = add_log_price(df)

    predictions = model.transform(df)

    if use_log_price:
        predictions = predictions.withColumn(
            "predicted_price", F.round(F.exp(F.col("prediction")), 0)
        )
    else:
        predictions = predictions.withColumn(
            "predicted_price", F.round(F.col("prediction"), 0)
        )

    return predictions


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def extract_feature_importance(
    model: PipelineModel,
    feature_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Extract GBT feature importances from a fitted PipelineModel.

    Parameters
    ----------
    model : PipelineModel
        Fitted pipeline that ends with a GBTRegressionModel.
    feature_names : list, optional
        Names corresponding to the feature vector positions.

    Returns
    -------
    dict mapping feature name → importance score.
    """
    gbt_model = None
    for stage in model.stages:
        if isinstance(stage, GBTRegressionModel):
            gbt_model = stage
            break

    if gbt_model is None:
        raise ValueError("No GBTRegressionModel found in the pipeline stages.")

    importances = gbt_model.featureImportances.toArray()

    if feature_names is None:
        feature_names = NUMERIC_FEATURES

    n = min(len(feature_names), len(importances))
    return {feature_names[i]: round(float(importances[i]), 4) for i in range(n)}


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: PipelineModel, path: str) -> None:
    """Save a fitted PipelineModel to disk."""
    model.write().overwrite().save(path)


def load_model(path: str) -> PipelineModel:
    """Load a saved PipelineModel from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return PipelineModel.load(path)
