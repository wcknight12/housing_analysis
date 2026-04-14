"""
PySpark MLlib feature engineering pipeline for Texas housing price prediction.

Transforms a listings DataFrame into a feature vector ready for MLlib estimators.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StandardScaler,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# Numeric features fed directly into the model
NUMERIC_FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "lot_size_acres",
    "age_years",
    "school_rating",
    "distance_from_center_miles",
    "property_tax_rate",
]

# Categorical features to one-hot-encode
CATEGORICAL_FEATURES = ["city"]

# Output column name for the assembled feature vector
FEATURES_COL = "features"
LABEL_COL = "price"


def build_feature_pipeline(
    numeric_features: list = None,
    categorical_features: list = None,
    scale: bool = True,
) -> Pipeline:
    """
    Build a PySpark ML Pipeline that:
      1. Imputes missing numeric values (median strategy)
      2. Indexes + one-hot encodes categorical features
      3. Assembles all features into a single vector
      4. Optionally scales numeric features (StandardScaler)

    Parameters
    ----------
    numeric_features : list, optional
        Names of numeric input columns (default: NUMERIC_FEATURES).
    categorical_features : list, optional
        Names of categorical input columns (default: CATEGORICAL_FEATURES).
    scale : bool
        Whether to apply StandardScaler to the final feature vector.

    Returns
    -------
    pyspark.ml.Pipeline
        Unfitted feature pipeline.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    stages = []

    # ── 1. Impute numeric columns ──────────────────────────────────────────
    imputer = Imputer(
        inputCols=numeric_features,
        outputCols=[f"{c}_imputed" for c in numeric_features],
        strategy="median",
    )
    stages.append(imputer)
    imputed_numeric = [f"{c}_imputed" for c in numeric_features]

    # ── 2. Index + OHE categorical columns ────────────────────────────────
    ohe_output_cols = []
    for cat in categorical_features:
        idx_col = f"{cat}_idx"
        ohe_col = f"{cat}_ohe"
        stages.append(StringIndexer(inputCol=cat, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=idx_col, outputCol=ohe_col))
        ohe_output_cols.append(ohe_col)

    # ── 3. Assemble all features into one vector ───────────────────────────
    assemble_cols = imputed_numeric + ohe_output_cols
    assembler = VectorAssembler(inputCols=assemble_cols, outputCol="raw_features")
    stages.append(assembler)

    # ── 4. Optional standard scaling ──────────────────────────────────────
    if scale:
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol=FEATURES_COL,
            withMean=False,
            withStd=True,
        )
        stages.append(scaler)
    else:
        stages.append(
            VectorAssembler(inputCols=["raw_features"], outputCol=FEATURES_COL)
        )

    return Pipeline(stages=stages)


def add_log_price(df: DataFrame, label_col: str = LABEL_COL) -> DataFrame:
    """Add a log-transformed label column (log_price) to reduce price skew."""
    return df.withColumn(f"log_{label_col}", F.log(F.col(label_col).cast("double")))


def add_price_per_sqft(df: DataFrame) -> DataFrame:
    """Ensure price_per_sqft column exists."""
    if "price_per_sqft" not in df.columns:
        df = df.withColumn(
            "price_per_sqft",
            F.round(F.col("price").cast("double") / F.col("sqft").cast("double"), 2),
        )
    return df
