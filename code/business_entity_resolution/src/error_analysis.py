from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_pair_features


ERROR_REPORT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "error_type",
    "secondary_error_type",
    "ambiguity_flag",
    "match_probability",
    "ground_truth_label",
    "name_exact_match",
    "name_token_jaccard",
    "name_char_similarity",
    "name_tfidf_similarity",
    "address_exact_match",
    "address_token_jaccard",
    "address_char_similarity",
    "country_exact_match",
    "shared_name_token_count",
    "shared_address_token_count",
    "shared_digit_count",
    "name_length_difference",
    "address_length_difference",
    "address_missing_target",
    "same_source_indicator",
]
SUMMARY_COLUMNS = ["error_type", "count"]


def classify_error_rows(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    threshold: float = 0.45,
    ambiguity_margin: float = 0.05,
    weak_similarity_threshold: float = 0.50,
) -> pd.DataFrame:
    """Classify persisted candidate rows using existing pairwise features."""
    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "match_probability",
        "ground_truth_label",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    merged = predictions.merge(
        features,
        on=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    if merged["name_exact_match"].isna().any():
        raise ValueError("features do not cover all persisted prediction rows")

    merged["predicted_match"] = merged["match_probability"] >= threshold
    merged["error_type"] = "negative_unselected"
    merged.loc[merged["predicted_match"] & merged["ground_truth_label"].eq(1), "error_type"] = "true_positive"
    merged.loc[merged["predicted_match"] & merged["ground_truth_label"].eq(0), "error_type"] = "false_positive"
    merged.loc[~merged["predicted_match"] & merged["ground_truth_label"].eq(1), "error_type"] = "false_negative"

    top_probability = merged.groupby("source1_entity_id")["match_probability"].transform("max")
    ambiguous = (
        merged["predicted_match"]
        & merged["match_probability"].ge(top_probability - ambiguity_margin)
        & merged.groupby("source1_entity_id")["predicted_match"].transform("sum").gt(1)
    )
    merged["ambiguity_flag"] = ambiguous.astype(np.int8)
    merged["secondary_error_type"] = ""

    error_rows = merged[
        merged["error_type"].isin(
            ["true_positive", "false_positive", "false_negative", "ambiguous"]
        )
    ].copy()
    exact_error = (
        error_rows["error_type"].isin(["false_positive", "false_negative"])
        & (
            error_rows["name_exact_match"].eq(1)
            | error_rows["address_exact_match"].eq(1)
        )
    )
    error_rows.loc[exact_error, "secondary_error_type"] = "exact_match_error"
    weak_error = (
        error_rows["name_char_similarity"].lt(weak_similarity_threshold)
        & error_rows["address_char_similarity"].lt(weak_similarity_threshold)
    )
    error_rows.loc[weak_error & ~exact_error, "secondary_error_type"] = "weak_similarity_error"
    return error_rows.reindex(columns=ERROR_REPORT_COLUMNS)


def build_error_summary(error_rows: pd.DataFrame) -> pd.DataFrame:
    """Count diagnostic error categories, including zero-count categories."""
    categories = [
        "true_positive",
        "false_positive",
        "false_negative",
        "ambiguous",
        "exact_match_error",
        "weak_similarity_error",
    ]
    counts = error_rows["error_type"].value_counts() if not error_rows.empty else pd.Series(dtype="int64")
    ambiguous_count = int(error_rows["ambiguity_flag"].sum()) if not error_rows.empty else 0
    return pd.DataFrame(
        {
            "error_type": categories,
            "count": [
                int(counts.get("true_positive", 0)),
                int(counts.get("false_positive", 0)),
                int(counts.get("false_negative", 0)),
                ambiguous_count,
                int((error_rows["secondary_error_type"] == "exact_match_error").sum()) if not error_rows.empty else 0,
                int((error_rows["secondary_error_type"] == "weak_similarity_error").sum()) if not error_rows.empty else 0,
            ],
        }
    )


def load_records_by_ids(path: str | Path, entity_ids: set[str], chunk_size: int = 250_000) -> pd.DataFrame:
    """Stream a training source TSV and retain only requested entity IDs."""
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
        matches = chunk[chunk["entity_id"].isin(entity_ids)]
        if not matches.empty:
            selected.append(matches)
        if sum(len(frame) for frame in selected) >= len(entity_ids):
            break
    if not selected:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    return pd.concat(selected, ignore_index=True).drop_duplicates("entity_id")


def run_error_analysis(
    predictions_path: str | Path,
    train_dir: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate error reports from persisted validation predictions and train sources."""
    predictions = pd.read_csv(predictions_path, sep="\t", dtype={"match_probability": float, "ground_truth_label": int})
    source1_ids = set(predictions["source1_entity_id"].astype(str))
    source2_ids = set(predictions.loc[predictions["candidate_source"].eq("S2"), "candidate_entity_id"].astype(str))
    source3_ids = set(predictions.loc[predictions["candidate_source"].eq("S3"), "candidate_entity_id"].astype(str))
    train_dir = Path(train_dir)
    source1 = load_records_by_ids(train_dir / "train_source1.tsv", source1_ids)
    source2 = load_records_by_ids(train_dir / "train_source2.tsv", source2_ids)
    source3 = load_records_by_ids(train_dir / "train_source3.tsv", source3_ids)
    features = build_pair_features(predictions[["source1_entity_id", "candidate_entity_id", "candidate_source"]], source1, source2, source3)
    errors = classify_error_rows(predictions, features, threshold=threshold)
    summary = build_error_summary(errors)
    report_path = Path(report_path)
    summary_path = Path(summary_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(report_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    return errors, summary
