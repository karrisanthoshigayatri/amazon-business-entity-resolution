"""Step 14 – Full-validation candidate-recall audit.

Runs candidate generation on the validation Source 1 entities with NO
target-row truncation (FULL_VALIDATION_CONFIG), then compares the result
against the bounded artifact produced in previous steps.

Usage (from the repo root)::

    python -m code.business_entity_resolution.src.run_full_validation_candidate_recall

Or from the code/business_entity_resolution directory::

    python -m src.run_full_validation_candidate_recall

Outputs (written to <repo_root>/output/):
    full_validation_candidate_recall_summary.tsv
    full_validation_candidate_recall_analysis.tsv
    bounded_vs_full_recall_comparison.tsv
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path gymnastics so the script can be run from multiple working directories
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
# Walk up until we find the repo root (contains "dataset/" and "output/")
_REPO_ROOT = _HERE
for _ in range(6):
    if (_REPO_ROOT / "dataset").is_dir() and (_REPO_ROOT / "output").is_dir():
        break
    _REPO_ROOT = _REPO_ROOT.parent

_SRC = _HERE.parent
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src.blocking import (  # noqa: E402  (after sys.path fix)
    FULL_VALIDATION_CONFIG,
    blocking_mode,
    generate_candidates,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR = _REPO_ROOT / "dataset" / "train"
OUTPUT_DIR = _REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.tsv"
BOUNDED_RECALL_SUMMARY_PATH = OUTPUT_DIR / "candidate_recall_summary.tsv"

FULL_SUMMARY_OUT = OUTPUT_DIR / "full_validation_candidate_recall_summary.tsv"
FULL_ANALYSIS_OUT = OUTPUT_DIR / "full_validation_candidate_recall_analysis.tsv"
COMPARISON_OUT = OUTPUT_DIR / "bounded_vs_full_recall_comparison.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_true_matches(ground_truth: pd.DataFrame) -> int:
    """Count individual target IDs across all matched_entity_ids entries."""
    return sum(
        len([x for x in str(v).split(",") if x])
        for v in ground_truth["matched_entity_ids"]
    )


def _truth_pairs(ground_truth: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for s1_id, matched in zip(
        ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]
    ):
        pairs.extend(
            (str(s1_id), tid)
            for tid in str(matched).split(",")
            if tid
        )
    return pairs


def _build_analysis(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-true-pair candidate status (PRESENT or MISSING)."""
    candidate_keys = set(
        zip(
            candidates["source1_entity_id"].astype(str),
            candidates["candidate_entity_id"].astype(str),
        )
    )
    rows = []
    for s1_id, true_id in _truth_pairs(ground_truth):
        status = (
            "CANDIDATE_PRESENT" if (s1_id, true_id) in candidate_keys
            else "CANDIDATE_MISSING"
        )
        rows.append(
            {
                "source1_entity_id": s1_id,
                "true_entity_id": true_id,
                "true_source": true_id[:2],
                "candidate_status": status,
            }
        )
    return pd.DataFrame(rows, columns=["source1_entity_id", "true_entity_id", "true_source", "candidate_status"])


def _load_bounded_values() -> dict[str, float]:
    """Load the previously-computed bounded recall metrics."""
    if not BOUNDED_RECALL_SUMMARY_PATH.exists():
        return {}
    df = pd.read_csv(BOUNDED_RECALL_SUMMARY_PATH, sep="\t")
    return dict(zip(df["metric"].astype(str), df["value"].astype(float)))


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_full_validation_audit() -> None:
    mode = blocking_mode(FULL_VALIDATION_CONFIG)
    print(f"[Step 14] Blocking mode: {mode}")
    assert mode == "FULL", f"Expected FULL mode, got {mode}"

    # ------------------------------------------------------------------
    # 1. Load validation S1 IDs from the persisted artifact
    # ------------------------------------------------------------------
    print("[Step 14] Loading validation S1 IDs from persisted artifact …")
    if not VALIDATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Validation artifact not found: {VALIDATION_PREDICTIONS_PATH}\n"
            "Run the model training step first."
        )
    artifact = pd.read_csv(VALIDATION_PREDICTIONS_PATH, sep="\t", dtype=str)
    validation_s1_ids: set[str] = set(artifact["source1_entity_id"].astype(str).unique())
    validation_source1_count = len(validation_s1_ids)
    print(f"[Step 14] Validation S1 entities: {validation_source1_count}")

    # ------------------------------------------------------------------
    # 2. Load training data (train only — no test data)
    # ------------------------------------------------------------------
    print("[Step 14] Loading training source files …")
    source1_full = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    source2_full = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    source3_full = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str).fillna("")
    ground_truth_full = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id", "matched_entity_ids"],
        keep_default_na=False,
    )

    print(f"[Step 14] train_source1 rows : {len(source1_full):,}")
    print(f"[Step 14] train_source2 rows : {len(source2_full):,}")
    print(f"[Step 14] train_source3 rows : {len(source3_full):,}")

    # ------------------------------------------------------------------
    # 3. Restrict Source 1 to the validation split only
    # ------------------------------------------------------------------
    source1_val = source1_full[
        source1_full["entity_id"].astype(str).isin(validation_s1_ids)
    ].reset_index(drop=True)
    print(f"[Step 14] Validation S1 rows after filter: {len(source1_val):,}")

    # ------------------------------------------------------------------
    # 4. Load ground truth for the validation S1 entities
    # ------------------------------------------------------------------
    ground_truth_val = ground_truth_full[
        ground_truth_full["source1_entity_id"].isin(validation_s1_ids)
    ].reset_index(drop=True)
    total_ground_truth_matches = _count_true_matches(ground_truth_val)
    print(f"[Step 14] Ground-truth matches for validation S1: {total_ground_truth_matches:,}")

    # ------------------------------------------------------------------
    # 5. Run FULL candidate generation (no target-row truncation)
    # ------------------------------------------------------------------
    print(f"[Step 14] Running candidate generation in {mode} mode …")
    print(f"[Step 14]   approximate_target_rows = {FULL_VALIDATION_CONFIG.approximate_target_rows!r}  (None = no truncation)")

    tracemalloc.start()
    t0 = time.perf_counter()

    candidates, stats = generate_candidates(
        source1_val,
        source2_full,
        source3_full,
        FULL_VALIDATION_CONFIG,
    )

    runtime = time.perf_counter() - t0
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)

    print(f"[Step 14] Candidate generation complete in {runtime:.1f}s  (peak memory: {peak_memory_mb:.1f} MB)")
    print(f"[Step 14] Total candidate pairs generated: {len(candidates):,}")

    # ------------------------------------------------------------------
    # 6. Compute recall metrics
    # ------------------------------------------------------------------
    analysis = _build_analysis(candidates, ground_truth_val)

    candidate_present = int((analysis["candidate_status"] == "CANDIDATE_PRESENT").sum())
    candidate_missing = int((analysis["candidate_status"] == "CANDIDATE_MISSING").sum())
    candidate_generation_recall = (
        candidate_present / total_ground_truth_matches
        if total_ground_truth_matches > 0
        else 0.0
    )

    # Per-S1 candidate counts
    per_s1_counts = (
        candidates.groupby("source1_entity_id").size()
        .reindex(list(validation_s1_ids), fill_value=0)
    )
    avg_candidates = float(per_s1_counts.mean())
    median_candidates = float(per_s1_counts.median())
    max_candidates = int(per_s1_counts.max())

    # candidate_reduction_ratio: full pairs / bounded pairs
    bounded_pair_count = len(artifact)  # the bounded artifact pair count
    candidate_reduction_ratio = (
        len(candidates) / bounded_pair_count if bounded_pair_count > 0 else float("nan")
    )

    print(f"[Step 14] candidate_present            : {candidate_present:,}")
    print(f"[Step 14] candidate_missing            : {candidate_missing:,}")
    print(f"[Step 14] candidate_generation_recall  : {candidate_generation_recall:.6f}")
    print(f"[Step 14] average_candidates_per_s1    : {avg_candidates:.2f}")

    # ------------------------------------------------------------------
    # 7. Write output/full_validation_candidate_recall_summary.tsv
    # ------------------------------------------------------------------
    summary_rows = [
        ("run_mode", mode),
        ("validation_source1_count", validation_source1_count),
        ("total_ground_truth_matches", total_ground_truth_matches),
        ("candidate_present_matches", candidate_present),
        ("candidate_missing_matches", candidate_missing),
        ("candidate_generation_recall", round(candidate_generation_recall, 10)),
        ("average_candidates_per_source1", round(avg_candidates, 4)),
        ("median_candidates_per_source1", round(median_candidates, 4)),
        ("maximum_candidates_per_source1", max_candidates),
        ("candidate_reduction_ratio", round(candidate_reduction_ratio, 4)),
        ("runtime_seconds", round(runtime, 2)),
        ("peak_memory_mb", round(peak_memory_mb, 2)),
        ("target_row_truncation", "NONE (approximate_target_rows=None)"),
        ("train_source2_total_rows", len(source2_full)),
        ("train_source3_total_rows", len(source3_full)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_df.to_csv(FULL_SUMMARY_OUT, sep="\t", index=False)
    print(f"[Step 14] Written: {FULL_SUMMARY_OUT}")

    # ------------------------------------------------------------------
    # 8. Write output/full_validation_candidate_recall_analysis.tsv
    # ------------------------------------------------------------------
    analysis.to_csv(FULL_ANALYSIS_OUT, sep="\t", index=False)
    print(f"[Step 14] Written: {FULL_ANALYSIS_OUT}")

    # ------------------------------------------------------------------
    # 9. Write output/bounded_vs_full_recall_comparison.tsv
    # ------------------------------------------------------------------
    bounded = _load_bounded_values()

    # Helper to safely get a bounded value
    def bv(key: str, fallback: object = "N/A") -> object:
        return bounded.get(key, fallback)

    def _change(b: object, f: object) -> str:
        try:
            diff = float(f) - float(b)  # type: ignore[arg-type]
            return f"{diff:+.6g}"
        except (TypeError, ValueError):
            return "N/A"

    comparison_rows = [
        (
            "run_mode",
            "BOUNDED(250000)",
            mode,
            "N/A",
        ),
        (
            "validation_source1_count",
            bv("N/A", validation_source1_count),   # bounded used same population
            validation_source1_count,
            "0",
        ),
        (
            "total_ground_truth_matches",
            bv("total_ground_truth_matches", "N/A"),
            total_ground_truth_matches,
            _change(bv("total_ground_truth_matches"), total_ground_truth_matches),
        ),
        (
            "candidate_present_matches",
            bv("candidate_present_matches", "N/A"),
            candidate_present,
            _change(bv("candidate_present_matches"), candidate_present),
        ),
        (
            "candidate_missing_matches",
            bv("candidate_missing_matches", "N/A"),
            candidate_missing,
            _change(bv("candidate_missing_matches"), candidate_missing),
        ),
        (
            "candidate_generation_recall",
            bv("candidate_generation_recall", "N/A"),
            round(candidate_generation_recall, 10),
            _change(bv("candidate_generation_recall"), candidate_generation_recall),
        ),
        (
            "average_candidates_per_source1",
            round(bounded_pair_count / validation_source1_count, 4) if validation_source1_count else "N/A",
            round(avg_candidates, 4),
            _change(
                bounded_pair_count / validation_source1_count if validation_source1_count else None,
                avg_candidates,
            ),
        ),
        (
            "median_candidates_per_source1",
            "N/A",   # not stored in bounded summary
            round(median_candidates, 4),
            "N/A",
        ),
        (
            "maximum_candidates_per_source1",
            "N/A",   # not stored in bounded summary
            max_candidates,
            "N/A",
        ),
        (
            "candidate_reduction_ratio",
            "1.0 (reference)",
            round(candidate_reduction_ratio, 4),
            _change(1.0, candidate_reduction_ratio),
        ),
        (
            "runtime",
            "N/A",
            f"{runtime:.2f}s",
            "N/A",
        ),
    ]

    comparison_df = pd.DataFrame(
        comparison_rows,
        columns=["metric", "bounded_value", "full_value", "change"],
    )
    comparison_df.to_csv(COMPARISON_OUT, sep="\t", index=False)
    print(f"[Step 14] Written: {COMPARISON_OUT}")

    print("\n[Step 14] ── Summary ──────────────────────────────────────────")
    print(f"  Mode                        : {mode}")
    print(f"  Validation S1 count         : {validation_source1_count:,}")
    print(f"  Total GT matches            : {total_ground_truth_matches:,}")
    print(f"  Candidate-present (FULL)    : {candidate_present:,}")
    print(f"  Candidate-missing (FULL)    : {candidate_missing:,}")
    print(f"  Recall (FULL)               : {candidate_generation_recall:.6f}")
    print(f"  Recall (BOUNDED, ref)       : {bv('candidate_generation_recall', 'N/A')}")
    print(f"  Candidate pairs (FULL)      : {len(candidates):,}")
    print(f"  Candidate pairs (BOUNDED)   : {bounded_pair_count:,}")
    print(f"  Runtime                     : {runtime:.1f}s")
    print(f"  Peak memory                 : {peak_memory_mb:.1f} MB")
    print("[Step 14] ─────────────────────────────────────────────────────")


if __name__ == "__main__":
    run_full_validation_audit()
