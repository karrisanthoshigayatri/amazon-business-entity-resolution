from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

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
    sample_size: int = 10_000
    ngram_size: int = 3
    max_postings_per_ngram: int = 2_000
    max_approx_candidates_per_query: int = 500
    approximate_target_rows: int | None = 250_000
    max_exact_block_size: int | None = None
    approximate_max_target_rows: int | None = None


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "source", "strategy"])


def _prepare(source1_df: pd.DataFrame, source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"entity_id", "business_name", "business_address", "country"}
    for label, frame in (("Source 1", source1_df), ("candidate source", source_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    left = pd.DataFrame({
        "source1_entity_id": source1_df["entity_id"].astype("string").reset_index(drop=True),
        "name": normalize_business_name_series(source1_df["business_name"]).reset_index(drop=True),
        "address": normalize_business_address_series(source1_df["business_address"]).reset_index(drop=True),
        "country": normalize_country_series(source1_df["country"]).reset_index(drop=True),
    })
    right = pd.DataFrame({
        "candidate_entity_id": source_df["entity_id"].astype("string").reset_index(drop=True),
        "name": normalize_business_name_series(source_df["business_name"]).reset_index(drop=True),
        "address": normalize_business_address_series(source_df["business_address"]).reset_index(drop=True),
        "country": normalize_country_series(source_df["country"]).reset_index(drop=True),
    })
    return left, right


def _frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    if not rows:
        return _empty_candidates()
    frame = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "strategy"])
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
    for source1_id, country, value in zip(source1["source1_entity_id"], source1["country"], source1[field]):
        if not value:
            continue
        for target_index in index.get((country, value), []):
            rows.append((source1_id, source.iloc[target_index]["candidate_entity_id"], source_label, strategy))
    return _frame(rows)


def generate_exact_name_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, max_block_size: int | None = None) -> pd.DataFrame:
    return _exact_candidates(source1, source, source_label, "name", "exact_name", max_block_size)


def generate_exact_address_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, max_block_size: int | None = None) -> pd.DataFrame:
    return _exact_candidates(source1, source, source_label, "address", "exact_address", max_block_size)


def _ngrams(value: str, ngram_size: int) -> set[str]:
    padded = f"  {value}  "
    if len(padded) <= ngram_size:
        return {padded}
    return {padded[index:index + ngram_size] for index in range(len(padded) - ngram_size + 1)}


def _approximate_candidates(source1: pd.DataFrame, source: pd.DataFrame, source_label: str, field: str, strategy: str, top_k: int, config: BlockingConfig) -> pd.DataFrame:
    if top_k <= 0:
        return _empty_candidates()
    target = source if config.approximate_target_rows is None else source.head(config.approximate_target_rows)
    target_ids = target["candidate_entity_id"].to_numpy(dtype=object)
    target_countries = target["country"].to_numpy(dtype=object)
    target_values = target[field].to_numpy(dtype=object)
    query_ids = source1["source1_entity_id"].to_numpy(dtype=object)
    query_countries = source1["country"].to_numpy(dtype=object)
    query_values = source1[field].to_numpy(dtype=object)
    postings: dict[str, list[int]] = defaultdict(list)
    for row_number, value in enumerate(target_values):
        if not value:
            continue
        for gram in _ngrams(value, config.ngram_size):
            if len(postings[gram]) < config.max_postings_per_ngram:
                postings[gram].append(row_number)

    rows: list[tuple[str, str, str, str]] = []
    for start in range(0, len(source1), config.query_batch_size):
        end = min(start + config.query_batch_size, len(query_values))
        for query_index in range(start, end):
            value = query_values[query_index]
            if not value:
                continue
            candidate_indices: set[int] = set()
            for gram in _ngrams(value, config.ngram_size):
                candidate_indices.update(postings.get(gram, []))
                if len(candidate_indices) >= config.max_approx_candidates_per_query:
                    break
            if not candidate_indices:
                continue
            scored: list[tuple[float, int]] = []
            query_grams = _ngrams(value, config.ngram_size)
            for candidate_index in candidate_indices:
                if target_countries[candidate_index] != query_countries[query_index]:
                    continue
                target_grams = _ngrams(target_values[candidate_index], config.ngram_size)
                union = len(query_grams | target_grams)
                score = len(query_grams & target_grams) / union if union else 0.0
                scored.append((score, candidate_index))
            scored.sort(reverse=True)
            for _, candidate_index in scored[:top_k]:
                rows.append((query_ids[query_index], target_ids[candidate_index], source_label, strategy))
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
