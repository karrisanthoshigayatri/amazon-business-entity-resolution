from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalization import (
    normalize_business_address_series,
    normalize_business_name_series,
    normalize_country_series,
)


IDENTIFIER_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def _require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _safe_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    maximum = np.maximum(left, right)
    return np.divide(
        np.minimum(left, right),
        maximum,
        out=np.zeros(len(left), dtype=np.float32),
        where=maximum != 0,
    )


def _token_sets(values: pd.Series) -> list[frozenset[str]]:
    return [frozenset(value.split()) for value in values.fillna("").astype(str)]


def _token_features(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_sets = _token_sets(left)
    right_sets = _token_sets(right)
    shared = np.fromiter(
        (len(left_tokens & right_tokens) for left_tokens, right_tokens in zip(left_sets, right_sets)),
        dtype=np.int32,
        count=len(left_sets),
    )
    unions = np.fromiter(
        (len(left_tokens | right_tokens) for left_tokens, right_tokens in zip(left_sets, right_sets)),
        dtype=np.int32,
        count=len(left_sets),
    )
    jaccard = np.divide(
        shared,
        unions,
        out=np.zeros(len(shared), dtype=np.float32),
        where=unions != 0,
    )
    return shared, jaccard, unions


def _char_similarity(left: pd.Series, right: pd.Series) -> np.ndarray:
    return np.fromiter(
        (
            SequenceMatcher(None, str(left_value), str(right_value)).ratio()
            for left_value, right_value in zip(left, right)
        ),
        dtype=np.float32,
        count=len(left),
    )


def _digit_sets(values: pd.Series) -> list[frozenset[str]]:
    return [frozenset(re.findall(r"\d", str(value))) for value in values.fillna("")]


def _shared_digit_count(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_digits = _digit_sets(left)
    right_digits = _digit_sets(right)
    return np.fromiter(
        (len(left_value & right_value) for left_value, right_value in zip(left_digits, right_digits)),
        dtype=np.int32,
        count=len(left_digits),
    )


def _hashed_tfidf_similarity(left: pd.Series, right: pd.Series) -> np.ndarray:
    values = pd.concat([left, right], ignore_index=True).fillna("").astype(str)
    inverse, unique_values = pd.factorize(values, sort=False)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=2**16,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    if not any(unique_values):
        return np.zeros(len(left), dtype=np.float32)
    matrix = vectorizer.fit_transform(unique_values.tolist())
    left_matrix = matrix[inverse[: len(left)]]
    right_matrix = matrix[inverse[len(left) :]]
    return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel().astype(np.float32)


def _prepare_records(source1_df: pd.DataFrame, source2_df: pd.DataFrame, source3_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["entity_id", "business_name", "business_address", "country"]
    for label, data in (("Source 1", source1_df), ("Source 2", source2_df), ("Source 3", source3_df)):
        _require_columns(data, required, label)

    source1 = pd.DataFrame(
        {
            "source1_entity_id": source1_df["entity_id"].astype(str),
            "s1_name": normalize_business_name_series(source1_df["business_name"]),
            "s1_address": normalize_business_address_series(source1_df["business_address"]),
            "s1_country": normalize_country_series(source1_df["country"]),
        }
    )
    targets = pd.concat(
        [
            source2_df.assign(_candidate_source="S2"),
            source3_df.assign(_candidate_source="S3"),
        ],
        ignore_index=True,
    )
    target = pd.DataFrame(
        {
            "candidate_entity_id": targets["entity_id"].astype(str),
            "candidate_source": targets["_candidate_source"].astype(str),
            "target_name": normalize_business_name_series(targets["business_name"]),
            "target_address": normalize_business_address_series(targets["business_address"]),
            "target_country": normalize_country_series(targets["country"]),
        }
    )
    return source1, target


def build_pair_features(
    candidates: pd.DataFrame,
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    include_tfidf: bool = True,
) -> pd.DataFrame:
    """Build pairwise features without using labels or ground truth."""
    _require_columns(candidates, IDENTIFIER_COLUMNS, "candidates")
    source1, targets = _prepare_records(source1_df, source2_df, source3_df)
    candidate_frame = candidates[IDENTIFIER_COLUMNS].copy()
    candidate_frame["candidate_source"] = candidate_frame["candidate_source"].astype(str)
    candidate_frame = candidate_frame.merge(source1, on="source1_entity_id", how="left", validate="many_to_one")
    candidate_frame = candidate_frame.merge(
        targets,
        on=["candidate_entity_id", "candidate_source"],
        how="left",
        validate="many_to_one",
    )
    if candidate_frame[["s1_name", "target_name"]].isna().any(axis=None):
        raise ValueError("Candidate IDs did not resolve to source records")

    name_shared, name_jaccard, _ = _token_features(candidate_frame["s1_name"], candidate_frame["target_name"])
    address_shared, address_jaccard, _ = _token_features(candidate_frame["s1_address"], candidate_frame["target_address"])
    s1_name_length = candidate_frame["s1_name"].str.len().to_numpy(dtype=np.float32)
    target_name_length = candidate_frame["target_name"].str.len().to_numpy(dtype=np.float32)
    s1_address_length = candidate_frame["s1_address"].str.len().to_numpy(dtype=np.float32)
    target_address_length = candidate_frame["target_address"].str.len().to_numpy(dtype=np.float32)

    features = candidate_frame[IDENTIFIER_COLUMNS].copy()
    features["name_exact_match"] = candidate_frame["s1_name"].eq(candidate_frame["target_name"]).astype(np.int8)
    features["name_token_jaccard"] = name_jaccard
    features["name_char_similarity"] = _char_similarity(candidate_frame["s1_name"], candidate_frame["target_name"])
    if include_tfidf:
        features["name_tfidf_similarity"] = _hashed_tfidf_similarity(candidate_frame["s1_name"], candidate_frame["target_name"])
    features["name_length_difference"] = np.abs(s1_name_length - target_name_length)
    features["name_length_ratio"] = _safe_ratio(s1_name_length, target_name_length)
    features["name_token_count_difference"] = np.abs(candidate_frame["s1_name"].str.split().str.len().to_numpy() - candidate_frame["target_name"].str.split().str.len().to_numpy())
    features["shared_name_token_count"] = name_shared
    features["address_exact_match"] = candidate_frame["s1_address"].eq(candidate_frame["target_address"]).astype(np.int8)
    features["address_token_jaccard"] = address_jaccard
    features["address_char_similarity"] = _char_similarity(candidate_frame["s1_address"], candidate_frame["target_address"])
    features["address_length_difference"] = np.abs(s1_address_length - target_address_length)
    features["address_length_ratio"] = _safe_ratio(s1_address_length, target_address_length)
    features["shared_address_token_count"] = address_shared
    features["shared_digit_count"] = _shared_digit_count(candidate_frame["s1_address"], candidate_frame["target_address"])
    features["country_exact_match"] = candidate_frame["s1_country"].eq(candidate_frame["target_country"]).astype(np.int8)
    features["name_missing_s1"] = candidate_frame["s1_name"].eq("").astype(np.int8)
    features["name_missing_target"] = candidate_frame["target_name"].eq("").astype(np.int8)
    features["address_missing_s1"] = candidate_frame["s1_address"].eq("").astype(np.int8)
    features["address_missing_target"] = candidate_frame["target_address"].eq("").astype(np.int8)
    features["same_source_indicator"] = features["candidate_source"].eq("S2").astype(np.int8)
    features["name_and_address_exact"] = (features["name_exact_match"] & features["address_exact_match"]).astype(np.int8)
    features["name_or_address_exact"] = (features["name_exact_match"] | features["address_exact_match"]).astype(np.int8)
    return features


def assign_training_labels(features_or_candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Append a binary training label using train ground truth only."""
    _require_columns(features_or_candidates, IDENTIFIER_COLUMNS, "features_or_candidates")
    _require_columns(ground_truth, ["source1_entity_id", "matched_entity_ids"], "ground_truth")
    true_pairs = {
        (str(source1_id), str(target_id))
        for source1_id, matched_ids in zip(
            ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]
        )
        for target_id in str(matched_ids).split(",")
        if target_id
    }
    result = features_or_candidates.copy()
    pair_keys = zip(result["source1_entity_id"].astype(str), result["candidate_entity_id"].astype(str))
    result["label"] = np.fromiter(
        ((source1_id, target_id) in true_pairs for source1_id, target_id in pair_keys),
        dtype=np.int8,
        count=len(result),
    )
    return result


# Short aliases for callers that prefer the stage terminology.
generate_pair_features = build_pair_features
assign_labels = assign_training_labels
