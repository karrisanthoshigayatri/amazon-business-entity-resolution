from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .normalization import (
    normalize_business_address_series,
    normalize_business_name_series,
    normalize_country_series,
)


@dataclass(frozen=True)
class BlockingConfig:
    name_top_k: int = 10
    address_top_k: int = 10
    query_batch_size: int = 5_000
    sample_size: int = 5_000
    ngram_size: int = 3
    ngram_features: int = 2**18
    approximate_target_rows: int | None = 250_000
    max_exact_block_size: int | None = None
    approximate_max_target_rows: int | None = None
    max_approx_candidates_per_query: int = 500


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "source",
            "strategy",
        ]
    )


def _prepare(source1_df: pd.DataFrame, source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"entity_id", "business_name", "business_address", "country"}
    for label, frame in (("Source 1", source1_df), ("candidate source", source_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    left = pd.DataFrame(
        {
            "source1_entity_id": source1_df["entity_id"].astype("string").reset_index(drop=True),
            "name": normalize_business_name_series(source1_df["business_name"]).reset_index(drop=True),
            "address": normalize_business_address_series(source1_df["business_address"]).reset_index(drop=True),
            "country": normalize_country_series(source1_df["country"]).reset_index(drop=True),
        }
    )
    right = pd.DataFrame(
        {
            "candidate_entity_id": source_df["entity_id"].astype("string").reset_index(drop=True),
            "name": normalize_business_name_series(source_df["business_name"]).reset_index(drop=True),
            "address": normalize_business_address_series(source_df["business_address"]).reset_index(drop=True),
            "country": normalize_country_series(source_df["country"]).reset_index(drop=True),
        }
    )
    return left, right


def _frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    if not rows:
        return _empty_candidates()
    frame = pd.DataFrame(
        rows,
        columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "strategy"],
    )
    frame["source"] = frame["candidate_source"]
    return frame[_empty_candidates().columns]


def _exact_index(source: pd.DataFrame, field: str, max_block_size: int | None) -> dict[tuple[str, str], list[int]]:
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_number, (country, value) in enumerate(zip(source["country"], source[field])):
        if not value:
            continue
        key = (country, value)
        if max_block_size is None or len(index[key]) < max_block_size:
            index[key].append(row_number)
    return index


def _exact_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, field: str, strategy: str, max_block_size: int | None) -> pd.DataFrame:
    index = _exact_index(source, field, max_block_size)
    rows: list[tuple[str, str, str, str]] = []
    target_ids = source["candidate_entity_id"].to_numpy(dtype=object)
    for source1_id, country, value in zip(source1["source1_entity_id"], source1["country"], source1[field]):
        if not value:
            continue
        rows.extend(
            (source1_id, target_ids[target_index], source_label, strategy)
            for target_index in index.get((country, value), [])
        )
    return _frame(rows)


def generate_exact_name_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, max_block_size: int | None = None) -> pd.DataFrame:
    return _exact_candidates(source1, source, source_label, "name", "exact_name", max_block_size)


def generate_exact_address_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, max_block_size: int | None = None) -> pd.DataFrame:
    return _exact_candidates(source1, source, source_label, "address", "exact_address", max_block_size)


def generate_ngram_ids(values: Iterable[str], ngram_size: int = 3, ngram_features: int = 2**18) -> sparse.csr_matrix:
    """Encode character n-grams as deterministic integer sparse feature IDs."""
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_size, ngram_size),
        n_features=ngram_features,
        binary=True,
        norm=None,
        alternate_sign=False,
        lowercase=False,
        dtype=np.float32,
    )
    return vectorizer.transform(pd.Series(values, dtype="string").fillna("")) .tocsr()


def intersect_candidate_indices(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sorted unique integer intersections without Python sets."""
    return np.intersect1d(left, right, assume_unique=False)


def lookup_postings(
    query_matrix: sparse.csr_matrix,
    target_matrix: sparse.csr_matrix,
    query_countries: np.ndarray,
    target_countries: np.ndarray,
    top_k: int,
    batch_size: int,
    max_candidates_per_query: int,
) -> list[np.ndarray]:
    """Retrieve top-k shared-ngram targets from sparse products, by country."""
    results: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(query_matrix.shape[0])]
    country_values = np.unique(query_countries)
    for country in country_values:
        if not country:
            continue
        query_indices = np.flatnonzero(query_countries == country)
        target_indices = np.flatnonzero(target_countries == country)
        if len(query_indices) == 0 or len(target_indices) == 0:
            continue
        country_target = target_matrix[target_indices]
        target_lengths = np.asarray(country_target.getnnz(axis=1)).ravel()
        for start in range(0, len(query_indices), batch_size):
            batch_indices = query_indices[start:start + batch_size]
            scores = (query_matrix[batch_indices] @ country_target.T).tocsr()
            query_lengths = np.asarray(query_matrix[batch_indices].getnnz(axis=1)).ravel()
            for row_number in range(scores.shape[0]):
                row = scores.getrow(row_number)
                if row.nnz == 0:
                    continue
                candidate_positions = row.indices
                intersections = row.data
                unions = query_lengths[row_number] + target_lengths[candidate_positions] - intersections
                similarities = intersections / np.maximum(unions, 1)
                limit = min(top_k, max_candidates_per_query, len(similarities))
                if limit < len(similarities):
                    chosen = np.argpartition(similarities, -limit)[-limit:]
                    chosen = chosen[np.argsort(similarities[chosen])[::-1]]
                else:
                    chosen = np.argsort(similarities)[::-1][:limit]
                results[batch_indices[row_number]] = target_indices[candidate_positions[chosen]].astype(np.int32)
    return results


def _approximate_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, field: str, strategy: str, top_k: int, config: BlockingConfig) -> pd.DataFrame:
    if top_k <= 0:
        return _empty_candidates()
    target = source if config.approximate_target_rows is None else source.head(config.approximate_target_rows)
    target_values = target[field].to_numpy(dtype=object)
    query_values = source1[field].to_numpy(dtype=object)
    target_matrix = generate_ngram_ids(target_values, config.ngram_size, config.ngram_features)
    query_matrix = generate_ngram_ids(query_values, config.ngram_size, config.ngram_features)
    target_ids = target["candidate_entity_id"].to_numpy(dtype=object)
    query_ids = source1["source1_entity_id"].to_numpy(dtype=object)
    target_countries = target["country"].to_numpy(dtype=object)
    query_countries = source1["country"].to_numpy(dtype=object)
    matched_indices = lookup_postings(
        query_matrix,
        target_matrix,
        query_countries,
        target_countries,
        top_k,
        config.query_batch_size,
        config.max_approx_candidates_per_query,
    )
    rows = [
        (query_ids[query_index], target_ids[target_index], source_label, strategy)
        for query_index, target_indices in enumerate(matched_indices)
        for target_index in target_indices
    ]
    return _frame(rows)


def generate_name_similarity_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, top_k: int = 10, config: BlockingConfig | None = None) -> pd.DataFrame:
    return _approximate_candidates(source1, source, source_label, "name", "name_similarity", top_k, config or BlockingConfig())


def generate_address_similarity_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, top_k: int = 10, config: BlockingConfig | None = None) -> pd.DataFrame:
    return _approximate_candidates(source1, source, source_label, "address", "address_similarity", top_k, config or BlockingConfig())


def combine_candidates(candidate_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in candidate_frames if not frame.empty]
    if not frames:
        return _empty_candidates()
    combined = pd.concat(frames, ignore_index=True)
    if "candidate_source" not in combined.columns and "source" in combined.columns:
        combined["candidate_source"] = combined["source"]
    if "source" not in combined.columns and "candidate_source" in combined.columns:
        combined["source"] = combined["candidate_source"]
    return combined.drop_duplicates(["source1_entity_id", "candidate_entity_id", "candidate_source"], keep="first").reset_index(drop=True)


def generate_candidates_for_source(source1_df: pd.DataFrame, source_df: pd.DataFrame, config: BlockingConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or BlockingConfig()
    prefixes = source_df["entity_id"].astype("string").str[:2].dropna().unique().tolist()
    if len(prefixes) != 1 or prefixes[0] not in {"S2", "S3"}:
        raise ValueError("Candidate source must contain only S2 or only S3 IDs")
    source_label = prefixes[0]
    source1, source = _prepare(source1_df, source_df)
    frames = [
        generate_exact_name_candidates(source1, source, source_label, config.max_exact_block_size),
        generate_exact_address_candidates(source1, source, source_label, config.max_exact_block_size),
        generate_name_similarity_candidates(source1, source, source_label, config.name_top_k, config),
        generate_address_similarity_candidates(source1, source, source_label, config.address_top_k, config),
    ]
    combined = combine_candidates(frames)
    stats = pd.DataFrame({"source1_entity_id": source1["source1_entity_id"]})
    for column, frame in zip(("exact_name_count", "exact_address_count", "name_similarity_count", "address_similarity_count"), frames):
        counts = frame.groupby("source1_entity_id").size() if not frame.empty else pd.Series(dtype="int64")
        stats[column] = stats["source1_entity_id"].map(counts).fillna(0).astype(int)
    final_counts = combined.groupby("source1_entity_id").size() if not combined.empty else pd.Series(dtype="int64")
    stats["final_candidate_count"] = stats["source1_entity_id"].map(final_counts).fillna(0).astype(int)
    return combined, stats


def generate_candidates(source1_df: pd.DataFrame, source2_df: pd.DataFrame, source3_df: pd.DataFrame, config: BlockingConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or BlockingConfig()
    s2_candidates, s2_stats = generate_candidates_for_source(source1_df, source2_df, config)
    s3_candidates, s3_stats = generate_candidates_for_source(source1_df, source3_df, config)
    return combine_candidates([s2_candidates, s3_candidates]), pd.concat([s2_stats, s3_stats], ignore_index=True)


def evaluate_candidate_recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> dict[str, object]:
    candidate_keys = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    totals: Counter[str] = Counter()
    found: Counter[str] = Counter()
    missed: list[tuple[str, str]] = []
    for source1_id, matched in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        for candidate_id in (item for item in str(matched).split(",") if item):
            source = candidate_id[:2]
            totals[source] += 1
            if (source1_id, candidate_id) in candidate_keys:
                found[source] += 1
            elif len(missed) < 20:
                missed.append((source1_id, candidate_id))
    total = sum(totals.values())
    return {
        "source2_true_matches": totals["S2"],
        "source2_recovered": found["S2"],
        "source3_true_matches": totals["S3"],
        "source3_recovered": found["S3"],
        "source2_recall": found["S2"] / totals["S2"] if totals["S2"] else 1.0,
        "source3_recall": found["S3"] / totals["S3"] if totals["S3"] else 1.0,
        "overall_recall": (found["S2"] + found["S3"]) / total if total else 1.0,
        "true_matches": total,
        "found_true_matches": found["S2"] + found["S3"],
        "missed_true_matches": total - found["S2"] - found["S3"],
        "missed_examples": missed,
    }


def build_candidate_pairs(source1_df, source2_df, source3_df, max_candidates_per_s1: int = 200):
    candidates, _ = generate_candidates(source1_df, source2_df, source3_df, BlockingConfig(max_exact_block_size=max_candidates_per_s1))
    return candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
