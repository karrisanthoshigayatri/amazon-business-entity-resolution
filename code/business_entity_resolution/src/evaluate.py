from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "match_probability",
}
REQUIRED_PERSISTED_COLUMNS = REQUIRED_PREDICTION_COLUMNS | {"ground_truth_label"}
REQUIRED_GROUND_TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}
DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.10, 1.00, 0.05))
THRESHOLD_RESULT_COLUMNS = [
    "threshold",
    "macro_precision",
    "macro_recall",
    "macro_f0_5",
    "total_TP",
    "total_FP",
    "total_FN",
]


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _match_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {item for item in str(value).split(",") if item}


def _metrics(true_matches: set[str], predicted_matches: set[str]) -> dict[str, Any]:
    true_positive = len(true_matches & predicted_matches)
    false_positive = len(predicted_matches - true_matches)
    false_negative = len(true_matches - predicted_matches)

    if not true_matches and not predicted_matches:
        precision = 1.0
        recall = 1.0
        f0_5 = 1.0
    else:
        precision = true_positive / len(predicted_matches) if predicted_matches else 0.0
        recall = true_positive / len(true_matches) if true_matches else 1.0
        denominator = 0.25 * precision + recall
        f0_5 = (1.25 * precision * recall / denominator) if denominator else 0.0

    return {
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5,
        "true_matches": sorted(true_matches),
        "predicted_matches": sorted(predicted_matches),
    }


def evaluate_predictions(
    validation_predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
    validation_source1_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate thresholded candidate probabilities with macro F0.5.

    Ground truth is used only for metric calculation. ``validation_source1_ids``
    should be supplied when validation entities with no candidate rows must be
    included in the macro average.
    """
    _require_columns(validation_predictions, REQUIRED_PREDICTION_COLUMNS, "validation_predictions")
    _require_columns(ground_truth, REQUIRED_GROUND_TRUTH_COLUMNS, "ground_truth")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    truth_by_source1 = {
        str(source1_id): _match_set(matches)
        for source1_id, matches in zip(
            ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]
        )
    }
    prediction_ids = validation_predictions.loc[
        validation_predictions["match_probability"] >= threshold
    ]
    predicted_by_source1 = {
        str(source1_id): set(group["candidate_entity_id"].astype(str))
        for source1_id, group in prediction_ids.groupby("source1_entity_id")
    }

    entity_ids = set(truth_by_source1) | set(predicted_by_source1)
    if validation_source1_ids is not None:
        entity_ids.update(str(entity_id) for entity_id in validation_source1_ids)

    rows: list[dict[str, Any]] = []
    for source1_id in sorted(entity_ids):
        row = _metrics(
            truth_by_source1.get(source1_id, set()),
            predicted_by_source1.get(source1_id, set()),
        )
        rows.append({"source1_entity_id": source1_id, **row})

    per_entity = pd.DataFrame(rows)
    if per_entity.empty:
        macro_precision = macro_recall = macro_f0_5 = 0.0
    else:
        macro_precision = float(per_entity["precision"].mean())
        macro_recall = float(per_entity["recall"].mean())
        macro_f0_5 = float(per_entity["f0_5"].mean())

    true_match_sets = [truth_by_source1.get(entity_id, set()) for entity_id in entity_ids]
    return {
        "threshold": threshold,
        "macro_f0_5": macro_f0_5,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_entity_metrics": per_entity,
        "false_positive_count": int(per_entity["false_positive_count"].sum()) if not per_entity.empty else 0,
        "false_negative_count": int(per_entity["false_negative_count"].sum()) if not per_entity.empty else 0,
        "true_positive_count": int(per_entity["true_positive_count"].sum()) if not per_entity.empty else 0,
        "validation_entity_count": len(entity_ids),
        "zero_match_count": sum(not matches for matches in true_match_sets),
        "singleton_count": sum(len(matches) == 1 for matches in true_match_sets),
        "correctly_empty_count": sum(
            not truth_by_source1.get(entity_id, set())
            and not predicted_by_source1.get(entity_id, set())
            for entity_id in entity_ids
        ),
        "minimum_entity_f0_5": float(per_entity["f0_5"].min()) if not per_entity.empty else 0.0,
        "maximum_entity_f0_5": float(per_entity["f0_5"].max()) if not per_entity.empty else 0.0,
        "mean_entity_f0_5": float(per_entity["f0_5"].mean()) if not per_entity.empty else 0.0,
    }


def evaluate_at_baseline_threshold(
    validation_predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    validation_source1_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the diagnostic 0.5 threshold; this does not claim optimality."""
    return evaluate_predictions(
        validation_predictions,
        ground_truth,
        threshold=0.5,
        validation_source1_ids=validation_source1_ids,
    )


def evaluate_persisted_validation_predictions(
    persisted_predictions: pd.DataFrame,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate a persisted validation artifact without retraining or extra data."""
    _require_columns(
        persisted_predictions,
        REQUIRED_PERSISTED_COLUMNS,
        "persisted_predictions",
    )
    validation_ids = persisted_predictions["source1_entity_id"].astype(str).unique()
    positive_rows = persisted_predictions[persisted_predictions["ground_truth_label"] == 1]
    ground_truth = (
        positive_rows.groupby("source1_entity_id")["candidate_entity_id"]
        .agg(lambda values: ",".join(sorted(set(values.astype(str)))))
        .reindex(validation_ids, fill_value="")
        .rename("matched_entity_ids")
        .reset_index()
    )
    return evaluate_predictions(
        persisted_predictions,
        ground_truth,
        threshold=threshold,
        validation_source1_ids=validation_ids,
    ) | {
        "evaluation_scope": "persisted_candidate_pairs_only",
        "warning": (
            "This artifact-only evaluation cannot count true matches that were "
            "missed during candidate generation or identify true zero-match "
            "entities absent from the persisted candidate pairs."
        ),
    }


def tune_persisted_thresholds(
    persisted_predictions: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    output_path: str | None = None,
    selected_threshold_path: str | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """Evaluate fixed thresholds and optionally persist the comparison table."""
    threshold_values = [float(value) for value in thresholds]
    if not threshold_values:
        raise ValueError("at least one threshold is required")
    rows = []
    for threshold in threshold_values:
        result = evaluate_persisted_validation_predictions(persisted_predictions, threshold)
        rows.append(
            {
                "threshold": threshold,
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "macro_f0_5": result["macro_f0_5"],
                "total_TP": result["true_positive_count"],
                "total_FP": result["false_positive_count"],
                "total_FN": result["false_negative_count"],
            }
        )
    comparison = pd.DataFrame(rows, columns=THRESHOLD_RESULT_COLUMNS)
    best_score = comparison["macro_f0_5"].max()
    tied = comparison.loc[
        np.isclose(comparison["macro_f0_5"], best_score, rtol=0, atol=1e-12),
        "threshold",
    ].tolist()
    if output_path is not None:
        comparison.to_csv(output_path, sep="\t", index=False)
    if selected_threshold_path is not None:
        with open(selected_threshold_path, "w", encoding="ascii", newline="\n") as handle:
            handle.write(f"{min(tied):.2f}\n")
    return comparison, tied
