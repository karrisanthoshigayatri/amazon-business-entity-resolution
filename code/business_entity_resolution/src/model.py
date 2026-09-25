from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from .features import IDENTIFIER_COLUMNS


LABEL_COLUMN = "label"
MODEL_RANDOM_STATE = 42


@dataclass
class BaselineModelResult:
    model: RandomForestClassifier
    feature_columns: list[str]
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    diagnostics: dict[str, Any]


def select_model_features(labeled_features: pd.DataFrame) -> list[str]:
    """Return numeric engineered columns, excluding IDs and the training label."""
    excluded = set(IDENTIFIER_COLUMNS) | {LABEL_COLUMN}
    return [
        column
        for column in labeled_features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(labeled_features[column])
    ]


def split_by_source1_entity(
    labeled_features: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = MODEL_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate pairs by Source 1 entity so groups cannot cross splits."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    groups = labeled_features["source1_entity_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_indices, validation_indices = next(splitter.split(labeled_features, groups=groups))
    return (
        labeled_features.iloc[train_indices].reset_index(drop=True),
        labeled_features.iloc[validation_indices].reset_index(drop=True),
    )


def train_baseline_model(
    labeled_features: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = MODEL_RANDOM_STATE,
    n_estimators: int = 100,
    max_depth: int | None = 12,
    n_jobs: int = -1,
) -> BaselineModelResult:
    """Train one reproducible class-weighted Random Forest baseline."""
    if LABEL_COLUMN not in labeled_features.columns:
        raise ValueError("labeled_features must contain a label column")
    feature_columns = select_model_features(labeled_features)
    if not feature_columns:
        raise ValueError("No numeric engineered feature columns were found")

    train_data, validation_data = split_by_source1_entity(
        labeled_features, validation_size, random_state
    )
    x_train = train_data[feature_columns].fillna(0).to_numpy(dtype=np.float32)
    y_train = train_data[LABEL_COLUMN].to_numpy(dtype=np.int8)
    x_validation = validation_data[feature_columns].fillna(0).to_numpy(dtype=np.float32)
    y_validation = validation_data[LABEL_COLUMN].to_numpy(dtype=np.int8)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=2,
    )
    train_started = time.perf_counter()
    model.fit(x_train, y_train)
    training_runtime = time.perf_counter() - train_started
    prediction_started = time.perf_counter()
    validation_probability = model.predict_proba(x_validation)[:, 1]
    prediction_runtime = time.perf_counter() - prediction_started

    validation_output = validation_data[IDENTIFIER_COLUMNS + [LABEL_COLUMN]].copy()
    validation_output["match_probability"] = validation_probability
    train_positive = int(y_train.sum())
    validation_positive = int(y_validation.sum())
    diagnostics = {
        "model_name": "RandomForestClassifier",
        "training_pairs": len(train_data),
        "validation_pairs": len(validation_data),
        "training_source1_entities": train_data["source1_entity_id"].nunique(),
        "validation_source1_entities": validation_data["source1_entity_id"].nunique(),
        "positive_training_pairs": train_positive,
        "negative_training_pairs": int(len(y_train) - train_positive),
        "positive_validation_pairs": validation_positive,
        "negative_validation_pairs": int(len(y_validation) - validation_positive),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "class_weight": "balanced_subsample",
        "training_runtime_seconds": training_runtime,
        "prediction_runtime_seconds": prediction_runtime,
        "probability_statistics": {
            "minimum": float(validation_probability.min()) if len(validation_probability) else 0.0,
            "maximum": float(validation_probability.max()) if len(validation_probability) else 0.0,
            "mean": float(validation_probability.mean()) if len(validation_probability) else 0.0,
            "median": float(np.median(validation_probability)) if len(validation_probability) else 0.0,
        },
        "warnings": [],
    }
    return BaselineModelResult(
        model=model,
        feature_columns=feature_columns,
        train_features=train_data,
        validation_features=validation_output,
        diagnostics=diagnostics,
    )


# Short aliases for callers that prefer stage terminology.
train_baseline = train_baseline_model
split_training_validation = split_by_source1_entity
