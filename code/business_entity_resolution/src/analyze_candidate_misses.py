"""Step 15 – Analyze the remaining 294 candidate-missing true matches.

For every ground-truth pair that was NOT recovered by full-validation
candidate generation, this module:
  1. Loads the raw source records for both sides of each missing pair.
  2. Computes normalized name / address / country and similarity scores.
  3. Categorizes each miss with a human-readable reason.
  4. Identifies which blocking strategy failed.
  5. Writes two output TSVs.

Usage (from repo root)::

    python -m code.business_entity_resolution.src.analyze_candidate_misses

Or from code/business_entity_resolution directory::

    python -m src.analyze_candidate_misses
"""
from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE
for _ in range(6):
    if (_REPO_ROOT / "dataset").is_dir() and (_REPO_ROOT / "output").is_dir():
        break
    _REPO_ROOT = _REPO_ROOT.parent

_SRC = _HERE.parent
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src.normalization import (  # noqa: E402
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)

TRAIN_DIR = _REPO_ROOT / "dataset" / "train"
OUTPUT_DIR = _REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_ANALYSIS_PATH = OUTPUT_DIR / "full_validation_candidate_recall_analysis.tsv"
MISSES_ANALYSIS_OUT = OUTPUT_DIR / "remaining_candidate_misses_analysis.tsv"
MISSES_SUMMARY_OUT = OUTPUT_DIR / "remaining_candidate_misses_summary.tsv"

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _char_sim(a: str, b: str) -> float:
    """Character-level SequenceMatcher ratio."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _digit_similarity(a: str, b: str) -> float:
    """Overlap of digit sequences between two strings."""
    da = re.findall(r"\d+", a)
    db = re.findall(r"\d+", b)
    if not da and not db:
        return 1.0
    if not da or not db:
        return 0.0
    set_a = set(da)
    set_b = set(db)
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Miss categorization
# ---------------------------------------------------------------------------

# Thresholds used to classify similarity as "weak"
_WEAK_SIM = 0.5       # character similarity below this → considered weak
_WEAK_TOKEN = 0.3     # token Jaccard below this → weak token overlap


def _categorize_miss(
    norm_name_s1: str,
    norm_name_true: str,
    norm_addr_s1: str,
    norm_addr_true: str,
    country_s1: str,
    country_true: str,
    name_sim: float,
    addr_sim: float,
) -> tuple[str, str]:
    """Return (candidate_missing_reason, failed_strategy).

    Both values are descriptive strings derived purely from the field values
    — no external data, no inference beyond what the record pair provides.
    """

    # --- Country ---
    if country_s1 and country_true and country_s1 != country_true:
        return "country_mismatch", "exact_name_blocking|exact_address_blocking|name_similarity|address_similarity"

    # --- Missing fields ---
    name_s1_missing = not norm_name_s1
    name_true_missing = not norm_name_true
    addr_s1_missing = not norm_addr_s1
    addr_true_missing = not norm_addr_true

    if name_s1_missing and addr_s1_missing:
        return "missing_both_fields_s1", "all_strategies"
    if name_true_missing and addr_true_missing:
        return "missing_both_fields_target", "all_strategies"
    if name_s1_missing or name_true_missing:
        # Only address-based blocking could have fired; check if address is also weak
        if addr_sim < _WEAK_SIM:
            return "missing_name_and_weak_address", "exact_name_blocking|name_similarity|exact_address_blocking|address_similarity"
        return "missing_name_field", "exact_name_blocking|name_similarity"
    if addr_s1_missing or addr_true_missing:
        if name_sim < _WEAK_SIM:
            return "missing_address_and_weak_name", "exact_address_blocking|address_similarity|exact_name_blocking|name_similarity"
        return "missing_address_field", "exact_address_blocking|address_similarity"

    # --- Exact match checks ---
    name_exact = norm_name_s1 == norm_name_true
    addr_exact = norm_addr_s1 == norm_addr_true

    # If both are exact, blocking SHOULD have found this pair — unexpected miss.
    if name_exact and addr_exact:
        return "unexpected_miss_both_exact", "unexpected_exact_block_failure"

    # --- Digit / address-number variation ---
    digits_s1 = re.findall(r"\d+", norm_addr_s1)
    digits_true = re.findall(r"\d+", norm_addr_true)
    digit_mismatch = bool(digits_s1) and bool(digits_true) and set(digits_s1) != set(digits_true)

    # --- Classify by dominant failure mode ---
    name_weak = name_sim < _WEAK_SIM
    addr_weak = addr_sim < _WEAK_SIM
    name_token_j = _token_jaccard(norm_name_s1, norm_name_true)
    addr_token_j = _token_jaccard(norm_addr_s1, norm_addr_true)

    if name_weak and addr_weak:
        reason = "both_name_and_address_weak"
        strategy = "exact_name_blocking|exact_address_blocking|name_similarity|address_similarity"
        return reason, strategy

    if not name_weak and addr_weak:
        # Name is similar enough — missed because address similarity was too low
        # for address-similarity blocking; name similarity should have found it.
        if digit_mismatch:
            reason = "address_digit_variation"
        elif addr_token_j < _WEAK_TOKEN:
            reason = "address_token_order_or_variation"
        else:
            reason = "address_similarity_below_top_k_threshold"
        return reason, "address_similarity|exact_address_blocking"

    if name_weak and not addr_weak:
        # Address is similar enough but name similarity was too low.
        if name_token_j >= 0.5:
            reason = "name_token_order_variation"
        else:
            # Check for common abbreviation/suffix patterns
            tokens_s1 = set(norm_name_s1.split())
            tokens_true = set(norm_name_true.split())
            shared = tokens_s1 & tokens_true
            if shared:
                reason = "name_abbreviation_or_suffix_variation"
            else:
                reason = "name_variation_no_shared_tokens"
        return reason, "name_similarity|exact_name_blocking"

    # Both similarities are moderate (≥ _WEAK_SIM) but match was still missed.
    # This means the pair scored in the top-K neighbourhood but wasn't selected,
    # OR the country/exact-match keys differed just enough.
    if name_sim >= 0.75 and addr_sim < 0.75:
        reason = "address_variation_near_threshold"
        strategy = "address_similarity|exact_address_blocking"
    elif addr_sim >= 0.75 and name_sim < 0.75:
        reason = "name_variation_near_threshold"
        strategy = "name_similarity|exact_name_blocking"
    elif name_sim >= 0.75 and addr_sim >= 0.75:
        reason = "near_match_outside_top_k"
        strategy = "name_similarity|address_similarity"
    else:
        reason = "moderate_similarity_below_retrieval_threshold"
        strategy = "name_similarity|address_similarity"

    return reason, strategy


# ---------------------------------------------------------------------------
# Record loader
# ---------------------------------------------------------------------------

def _load_id_lookup(path: Path, ids: set[str]) -> dict[str, dict[str, str]]:
    """Load only the rows whose entity_id is in `ids` — memory-efficient."""
    lookup: dict[str, dict[str, str]] = {}
    chunk_size = 100_000
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunk_size):
        chunk = chunk.fillna("")
        rows = chunk[chunk["entity_id"].isin(ids)]
        for row in rows.itertuples(index=False):
            lookup[row.entity_id] = {
                "entity_id": row.entity_id,
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }
        if len(lookup) == len(ids):
            break  # found all, stop early
    return lookup


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_candidate_misses(
    full_analysis_path: Path = FULL_ANALYSIS_PATH,
    train_dir: Path = TRAIN_DIR,
    misses_analysis_out: Path = MISSES_ANALYSIS_OUT,
    misses_summary_out: Path = MISSES_SUMMARY_OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze 294 candidate-missing pairs; write two output TSVs.

    Returns (analysis_df, summary_df).
    """
    # ------------------------------------------------------------------
    # 1. Load the full-validation analysis and filter to misses
    # ------------------------------------------------------------------
    full_df = pd.read_csv(full_analysis_path, sep="\t", dtype=str)
    misses = full_df[full_df["candidate_status"] == "CANDIDATE_MISSING"].reset_index(drop=True)
    total_true = len(full_df)
    candidate_present = int((full_df["candidate_status"] == "CANDIDATE_PRESENT").sum())
    candidate_missing = len(misses)
    print(f"[Step 15] Total true matches   : {total_true}")
    print(f"[Step 15] Candidate-present    : {candidate_present}")
    print(f"[Step 15] Candidate-missing    : {candidate_missing}")

    # ------------------------------------------------------------------
    # 2. Collect unique IDs to look up
    # ------------------------------------------------------------------
    s1_ids = set(misses["source1_entity_id"].astype(str).unique())
    s2_ids = set(misses.loc[misses["true_source"] == "S2", "true_entity_id"].astype(str).unique())
    s3_ids = set(misses.loc[misses["true_source"] == "S3", "true_entity_id"].astype(str).unique())

    print(f"[Step 15] Loading records for {len(s1_ids)} S1, {len(s2_ids)} S2, {len(s3_ids)} S3 entities …")

    s1_lookup = _load_id_lookup(train_dir / "train_source1.tsv", s1_ids)
    s2_lookup = _load_id_lookup(train_dir / "train_source2.tsv", s2_ids) if s2_ids else {}
    s3_lookup = _load_id_lookup(train_dir / "train_source3.tsv", s3_ids) if s3_ids else {}
    target_lookup = {**s2_lookup, **s3_lookup}

    print(f"[Step 15] Records loaded: {len(s1_lookup)} S1, {len(s2_lookup)} S2, {len(s3_lookup)} S3")

    # ------------------------------------------------------------------
    # 3. Build per-pair analysis rows
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for _, miss in misses.iterrows():
        s1_id = str(miss["source1_entity_id"])
        true_id = str(miss["true_entity_id"])
        true_source = str(miss["true_source"])

        s1_rec = s1_lookup.get(s1_id, {})
        tgt_rec = target_lookup.get(true_id, {})

        raw_name_s1 = s1_rec.get("business_name", "")
        raw_name_tgt = tgt_rec.get("business_name", "")
        raw_addr_s1 = s1_rec.get("business_address", "")
        raw_addr_tgt = tgt_rec.get("business_address", "")
        raw_country_s1 = s1_rec.get("country", "")
        raw_country_tgt = tgt_rec.get("country", "")

        norm_name_s1 = normalize_business_name(raw_name_s1)
        norm_name_tgt = normalize_business_name(raw_name_tgt)
        norm_addr_s1 = normalize_business_address(raw_addr_s1)
        norm_addr_tgt = normalize_business_address(raw_addr_tgt)
        country_s1 = normalize_country(raw_country_s1)
        country_tgt = normalize_country(raw_country_tgt)

        name_sim = round(_char_sim(norm_name_s1, norm_name_tgt), 6)
        addr_sim = round(_char_sim(norm_addr_s1, norm_addr_tgt), 6)
        digit_sim = round(_digit_similarity(norm_addr_s1, norm_addr_tgt), 6)

        reason, strategy = _categorize_miss(
            norm_name_s1, norm_name_tgt,
            norm_addr_s1, norm_addr_tgt,
            country_s1, country_tgt,
            name_sim, addr_sim,
        )

        rows.append({
            "source1_entity_id": s1_id,
            "true_entity_id": true_id,
            "true_source": true_source,
            "normalized_name_s1": norm_name_s1,
            "normalized_name_true": norm_name_tgt,
            "normalized_address_s1": norm_addr_s1,
            "normalized_address_true": norm_addr_tgt,
            "country_s1": country_s1,
            "country_true": country_tgt,
            "name_similarity": name_sim,
            "address_similarity": addr_sim,
            "digit_similarity": digit_sim,
            "candidate_missing_reason": reason,
            "failed_strategy": strategy,
        })

    analysis_df = pd.DataFrame(rows, columns=[
        "source1_entity_id", "true_entity_id", "true_source",
        "normalized_name_s1", "normalized_name_true",
        "normalized_address_s1", "normalized_address_true",
        "country_s1", "country_true",
        "name_similarity", "address_similarity", "digit_similarity",
        "candidate_missing_reason", "failed_strategy",
    ])

    # ------------------------------------------------------------------
    # 4. Build summary
    # ------------------------------------------------------------------
    recall = candidate_present / total_true if total_true else 0.0
    miss_rate = candidate_missing / total_true if total_true else 0.0

    # Source breakdown
    s2_miss = int((misses["true_source"] == "S2").sum())
    s3_miss = int((misses["true_source"] == "S3").sum())

    # Miss category counts
    reason_counts = analysis_df["candidate_missing_reason"].value_counts()
    # Failed strategy counts — a pair can list multiple strategies (pipe-separated);
    # count each strategy mention once per pair
    strategy_counts: dict[str, int] = {}
    for strats in analysis_df["failed_strategy"]:
        for s in str(strats).split("|"):
            s = s.strip()
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

    # Similarity statistics for missing pairs
    mean_name_sim = round(float(analysis_df["name_similarity"].mean()), 4)
    mean_addr_sim = round(float(analysis_df["address_similarity"].mean()), 4)
    name_exact_count = int((analysis_df["name_similarity"] == 1.0).sum())
    addr_exact_count = int((analysis_df["address_similarity"] == 1.0).sum())
    name_above_75 = int((analysis_df["name_similarity"] >= 0.75).sum())
    addr_above_75 = int((analysis_df["address_similarity"] >= 0.75).sum())

    summary_rows: list[tuple[str, Any]] = [
        ("total_true_matches", total_true),
        ("candidate_present", candidate_present),
        ("candidate_missing", candidate_missing),
        ("full_candidate_recall", round(recall, 10)),
        ("candidate_missing_rate", round(miss_rate, 10)),
        ("candidate_missing_s2", s2_miss),
        ("candidate_missing_s3", s3_miss),
        ("mean_name_similarity_missing", mean_name_sim),
        ("mean_address_similarity_missing", mean_addr_sim),
        ("name_exact_match_count_in_missing", name_exact_count),
        ("address_exact_match_count_in_missing", addr_exact_count),
        ("name_similarity_above_0_75_in_missing", name_above_75),
        ("address_similarity_above_0_75_in_missing", addr_above_75),
        ("", ""),  # blank separator
        ("--- miss_category_counts ---", ""),
    ]
    for cat, cnt in reason_counts.items():
        pct = round(100.0 * cnt / candidate_missing, 2)
        summary_rows.append((f"miss_category::{cat}", f"{cnt} ({pct}%)"))

    summary_rows.append(("", ""))
    summary_rows.append(("--- failed_strategy_counts ---", ""))
    for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = round(100.0 * cnt / candidate_missing, 2)
        summary_rows.append((f"failed_strategy::{strat}", f"{cnt} ({pct}%)"))

    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    analysis_df.to_csv(misses_analysis_out, sep="\t", index=False)
    summary_df.to_csv(misses_summary_out, sep="\t", index=False)
    print(f"[Step 15] Written: {misses_analysis_out}")
    print(f"[Step 15] Written: {misses_summary_out}")

    # ------------------------------------------------------------------
    # 6. Console summary
    # ------------------------------------------------------------------
    print("\n[Step 15] ── Miss Category Breakdown ─────────────────────────")
    for cat, cnt in reason_counts.items():
        pct = round(100.0 * cnt / candidate_missing, 2)
        print(f"  {cat:<45} {cnt:>4}  ({pct:>5}%)")
    print("\n[Step 15] ── Failed Strategy Breakdown ────────────────────────")
    for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = round(100.0 * cnt / candidate_missing, 2)
        print(f"  {strat:<45} {cnt:>4}  ({pct:>5}%)")
    print(f"\n[Step 15] S2 misses: {s2_miss}  |  S3 misses: {s3_miss}")
    print(f"[Step 15] Mean name sim (missing): {mean_name_sim}  |  Mean addr sim (missing): {mean_addr_sim}")
    print("[Step 15] ─────────────────────────────────────────────────────\n")

    return analysis_df, summary_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyze_candidate_misses()
