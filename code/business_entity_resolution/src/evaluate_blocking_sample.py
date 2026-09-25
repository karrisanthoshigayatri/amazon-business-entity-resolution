from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import pandas as pd

from .blocking import BlockingConfig, evaluate_candidate_recall, generate_candidates
from .normalization import (
    normalize_business_address_series,
    normalize_business_name_series,
    normalize_country_series,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
CHUNK_SIZE = 250_000
SAMPLE_SIZE = 5_000
DIAGNOSTIC_TARGET_ROWS = 25_000
GROUND_TRUTH_CHUNK_SIZE = 250_000


def _normalized_keys(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": normalize_business_name_series(data["business_name"]),
            "address": normalize_business_address_series(data["business_address"]),
            "country": normalize_country_series(data["country"]),
        },
        index=data.index,
    )


def load_diagnostic_target(
    path: Path,
    sample_source1: pd.DataFrame,
    approximate_target_rows: int,
    diagnostic_target_rows: int = DIAGNOSTIC_TARGET_ROWS,
) -> pd.DataFrame:
    """Load exact-key rows plus a bounded country-aware approximate pool.

    This is only for the diagnostic sample. Full candidate generation still
    receives the complete target source and uses its configured algorithm.
    """
    sample_keys = _normalized_keys(sample_source1)
    exact_names = set(zip(sample_keys["country"], sample_keys["name"])) - {("", "")}
    exact_addresses = set(zip(sample_keys["country"], sample_keys["address"])) - {("", "")}
    sample_countries = set(sample_keys["country"]) - {""}
    selected: list[pd.DataFrame] = []
    approximate_counts: dict[str, int] = {}

    rows_read = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        remaining_rows = diagnostic_target_rows - rows_read
        if remaining_rows <= 0:
            break
        chunk = chunk.head(remaining_rows)
        rows_read += len(chunk)
        keys = _normalized_keys(chunk)
        exact_mask = pd.Series(False, index=chunk.index)
        exact_mask |= pd.MultiIndex.from_arrays(
            [keys["country"], keys["name"]]
        ).isin(exact_names)
        exact_mask |= pd.MultiIndex.from_arrays(
            [keys["country"], keys["address"]]
        ).isin(exact_addresses)

        approximate_mask = keys["country"].isin(sample_countries)
        approximate_rows = chunk.loc[approximate_mask & ~exact_mask]
        keep_indices: list[int] = []
        for country, country_rows in approximate_rows.groupby(
            keys.loc[approximate_rows.index, "country"]
        ):
            remaining = max(approximate_target_rows - approximate_counts.get(country, 0), 0)
            chosen = country_rows.head(remaining)
            keep_indices.extend(chosen.index.tolist())
            approximate_counts[country] = approximate_counts.get(country, 0) + len(chosen)

        selected_chunk = chunk.loc[exact_mask | chunk.index.isin(keep_indices)]
        if not selected_chunk.empty:
            selected.append(selected_chunk)

    if not selected:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    return pd.concat(selected, ignore_index=True).drop_duplicates("entity_id")


def load_sample_ground_truth(sample_ids: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=GROUND_TRUTH_CHUNK_SIZE,
    ):
        selected = chunk[chunk["source1_entity_id"].isin(sample_ids)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    return pd.concat(chunks, ignore_index=True)


def run_sample(sample_size: int = SAMPLE_SIZE) -> dict[str, object]:
    started = time.time()
    tracemalloc.start()
    source1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        dtype=str,
        nrows=sample_size,
        keep_default_na=False,
    )
    source2 = load_diagnostic_target(TRAIN_DIR / "train_source2.tsv", source1, BlockingConfig().approximate_target_rows or 0)
    source3 = load_diagnostic_target(
        TRAIN_DIR / "train_source3.tsv", source1, BlockingConfig().approximate_target_rows or 0
    )
    sample_ids = set(source1["entity_id"])
    ground_truth = load_sample_ground_truth(sample_ids)

    config = BlockingConfig(
        name_top_k=10,
        address_top_k=10,
        query_batch_size=5_000,
        approximate_target_rows=None,
    )
    candidates, _ = generate_candidates(source1, source2, source3, config)
    recall = evaluate_candidate_recall(candidates, ground_truth)
    per_source1 = candidates.groupby("source1_entity_id").size().reindex(
        source1["entity_id"], fill_value=0
    )
    _, peak = tracemalloc.get_traced_memory()
    possible_pairs = len(source1) * (5_034_616 + 5_285_603)
    result = {
        "sample_source1": len(source1),
        "diagnostic_source2_rows": len(source2),
        "diagnostic_source3_rows": len(source3),
        "s2_candidates": int((candidates["candidate_source"] == "S2").sum()),
        "s3_candidates": int((candidates["candidate_source"] == "S3").sum()),
        "total_candidate_pairs": len(candidates),
        "average_candidates_per_s1": float(per_source1.mean()),
        "median_candidates_per_s1": float(per_source1.median()),
        "max_candidates_per_s1": int(per_source1.max()),
        "reduction_ratio": 1 - len(candidates) / possible_pairs,
        "ground_truth_rows": len(ground_truth),
        "recall": recall,
        "peak_tracemalloc_mb": peak / 1024 / 1024,
        "runtime_seconds": time.time() - started,
    }
    for key, value in result.items():
        print(f"{key}={value}")
    for key, value in recall.items():
        print(f"recall_{key}={value}")
    return result


if __name__ == "__main__":
    run_sample()
