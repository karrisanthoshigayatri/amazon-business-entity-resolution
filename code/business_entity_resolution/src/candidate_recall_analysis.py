from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .error_analysis import load_records_by_ids
from .features import build_pair_features
from .normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)


ANALYSIS_COLUMNS = [
    "source1_entity_id",
    "true_entity_id",
    "true_source",
    "candidate_status",
    "match_probability",
    "name_similarity_features",
    "address_similarity_features",
    "country_features",
]


def _truth_pairs(ground_truth: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = []
    for source1_id, matched_ids in zip(
        ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]
    ):
        pairs.extend(
            (str(source1_id), target_id)
            for target_id in str(matched_ids).split(",")
            if target_id
        )
    return pairs


def analyze_candidate_recall(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    pair_features: pd.DataFrame,
    record_lookup: dict[str, dict[str, Any]] | None = None,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify every validation true pair as candidate-present or missing."""
    candidate_keys = set(
        zip(predictions["source1_entity_id"].astype(str), predictions["candidate_entity_id"].astype(str))
    )
    prediction_lookup = {
        (str(row.source1_entity_id), str(row.candidate_entity_id)): row
        for row in predictions.itertuples()
    }
    feature_lookup = {
        (str(row.source1_entity_id), str(row.candidate_entity_id)): row
        for row in pair_features.itertuples()
    }
    rows = []
    for source1_id, true_id in _truth_pairs(ground_truth):
        key = (source1_id, true_id)
        source = true_id[:2]
        status = "CANDIDATE_PRESENT" if key in candidate_keys else "CANDIDATE_MISSING"
        prediction = prediction_lookup.get(key)
        feature_row = feature_lookup.get(key)
        probability = float(prediction.match_probability) if prediction is not None else None

        if feature_row is not None:
            name_features = {
                name: getattr(feature_row, name, None)
                for name in (
                    "name_exact_match", "name_token_jaccard", "name_char_similarity",
                    "name_tfidf_similarity", "name_length_difference", "name_length_ratio",
                    "shared_name_token_count",
                )
            }
            address_features = {
                name: getattr(feature_row, name, None)
                for name in (
                    "address_exact_match", "address_token_jaccard", "address_char_similarity",
                    "address_length_difference", "address_length_ratio", "shared_address_token_count",
                    "shared_digit_count", "address_missing_target",
                )
            }
            country_features = {"country_exact_match": getattr(feature_row, "country_exact_match", None)}
        else:
            source1_record = (record_lookup or {}).get(source1_id, {})
            target_record = (record_lookup or {}).get(true_id, {})
            source1_name = normalize_business_name(source1_record.get("business_name", ""))
            target_name = normalize_business_name(target_record.get("business_name", ""))
            source1_address = normalize_business_address(source1_record.get("business_address", ""))
            target_address = normalize_business_address(target_record.get("business_address", ""))
            name_features = {
                "normalized_name_s1": source1_name,
                "normalized_name_target": target_name,
                "name_exact": int(source1_name == target_name and bool(source1_name)),
            }
            address_features = {
                "normalized_address_s1": source1_address,
                "normalized_address_target": target_address,
                "address_exact": int(source1_address == target_address and bool(source1_address)),
                "address_missing_s1": int(not source1_address),
                "address_missing_target": int(not target_address),
            }
            country_features = {
                "country_s1": normalize_country(source1_record.get("country", "")),
                "country_target": normalize_country(target_record.get("country", "")),
                "country_exact": int(
                    normalize_country(source1_record.get("country", ""))
                    == normalize_country(target_record.get("country", ""))
                ),
            }

        rows.append(
            {
                "source1_entity_id": source1_id,
                "true_entity_id": true_id,
                "true_source": source,
                "candidate_status": status,
                "match_probability": probability,
                "name_similarity_features": json.dumps(name_features, sort_keys=True, default=float),
                "address_similarity_features": json.dumps(address_features, sort_keys=True, default=float),
                "country_features": json.dumps(country_features, sort_keys=True, default=float),
            }
        )

    analysis = pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)
    present = analysis[analysis["candidate_status"] == "CANDIDATE_PRESENT"]
    summary = pd.DataFrame(
        {
            "metric": [
                "total_ground_truth_matches",
                "candidate_present_matches",
                "candidate_missing_matches",
                "candidate_generation_recall",
                "candidate_present_above_threshold",
                "candidate_present_below_threshold",
            ],
            "value": [
                len(analysis),
                int(len(present)),
                int((analysis["candidate_status"] == "CANDIDATE_MISSING").sum()),
                len(present) / len(analysis) if len(analysis) else 0.0,
                int((present["match_probability"] >= threshold).sum()) if not present.empty else 0,
                int((present["match_probability"] < threshold).sum()) if not present.empty else 0,
            ],
        }
    )
    return analysis, summary


def run_candidate_recall_analysis(
    predictions_path: str | Path,
    train_dir: str | Path,
    analysis_path: str | Path,
    summary_path: str | Path,
    threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(predictions_path, sep="\t", dtype={"match_probability": float, "ground_truth_label": int})
    validation_ids = set(predictions["source1_entity_id"].astype(str))
    train_dir = Path(train_dir)
    ground_truth = pd.read_csv(
        train_dir / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id", "matched_entity_ids"],
        keep_default_na=False,
    )
    ground_truth = ground_truth[ground_truth["source1_entity_id"].isin(validation_ids)]
    candidate_source1_ids = validation_ids
    candidate_s2_ids = set(predictions.loc[predictions["candidate_source"].eq("S2"), "candidate_entity_id"].astype(str))
    candidate_s3_ids = set(predictions.loc[predictions["candidate_source"].eq("S3"), "candidate_entity_id"].astype(str))
    true_ids = {target_id for _, target_id in _truth_pairs(ground_truth)}
    source1 = load_records_by_ids(train_dir / "train_source1.tsv", candidate_source1_ids)
    source2 = load_records_by_ids(train_dir / "train_source2.tsv", candidate_s2_ids | {item for item in true_ids if item.startswith("S2-")})
    source3 = load_records_by_ids(train_dir / "train_source3.tsv", candidate_s3_ids | {item for item in true_ids if item.startswith("S3-")})
    lookup = {
        str(row.entity_id): row._asdict() for row in pd.concat([source1, source2, source3], ignore_index=True).itertuples()
    }
    pair_features = build_pair_features(
        predictions[["source1_entity_id", "candidate_entity_id", "candidate_source"]],
        source1,
        source2,
        source3,
    )
    analysis, summary = analyze_candidate_recall(
        predictions, ground_truth, pair_features, lookup, threshold
    )
    analysis_path = Path(analysis_path)
    summary_path = Path(summary_path)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(analysis_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    return analysis, summary
