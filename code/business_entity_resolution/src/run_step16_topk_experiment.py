"""Step 16 – Targeted top-K candidate-generation improvement experiment.

Baseline  : name_top_k=10, address_top_k=10  (FULL_VALIDATION_CONFIG)
Experiment: name_top_k=20, address_top_k=20  (2× baseline, approximate_target_rows=None)

Both runs use:
  - The same 1 000 validation S1 entities from validation_predictions.tsv
  - All train_source2 and train_source3 rows  (approximate_target_rows=None)
  - The same chunked similarity path  (target_chunk_size=500 000)
  - Only dataset/train/ — no test data
  - No model, feature, or threshold changes

Outputs:
  output/candidate_generation_step16_comparison.tsv
  output/candidate_generation_step16_miss_recovery.tsv
"""
from __future__ import annotations

import time
import tracemalloc
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
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

from src.blocking import (  # noqa: E402
    FULL_VALIDATION_CONFIG,
    BlockingConfig,
    blocking_mode,
    generate_candidates,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR   = _REPO_ROOT / "dataset" / "train"
OUTPUT_DIR  = _REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.tsv"
BASELINE_SUMMARY_PATH       = OUTPUT_DIR / "full_validation_candidate_recall_summary.tsv"
BASELINE_ANALYSIS_PATH      = OUTPUT_DIR / "full_validation_candidate_recall_analysis.tsv"
MISSES_ANALYSIS_PATH        = OUTPUT_DIR / "remaining_candidate_misses_analysis.tsv"

COMPARISON_OUT   = OUTPUT_DIR / "candidate_generation_step16_comparison.tsv"
RECOVERY_OUT     = OUTPUT_DIR / "candidate_generation_step16_miss_recovery.tsv"

# ---------------------------------------------------------------------------
# Experimental config (2× baseline K, full target, same chunking)
# ---------------------------------------------------------------------------
BASELINE_NAME_TOP_K    = FULL_VALIDATION_CONFIG.name_top_k       # 10
BASELINE_ADDRESS_TOP_K = FULL_VALIDATION_CONFIG.address_top_k    # 10

EXPERIMENTAL_NAME_TOP_K    = BASELINE_NAME_TOP_K * 2             # 20
EXPERIMENTAL_ADDRESS_TOP_K = BASELINE_ADDRESS_TOP_K * 2          # 20

EXPERIMENTAL_CONFIG = BlockingConfig(
    name_top_k=EXPERIMENTAL_NAME_TOP_K,
    address_top_k=EXPERIMENTAL_ADDRESS_TOP_K,
    approximate_target_rows=None,          # FULL – no truncation
    target_chunk_size=500_000,             # same chunking as baseline
    max_approx_candidates_per_query=500,   # unchanged
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_true_matches(ground_truth: pd.DataFrame) -> int:
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
            (str(s1_id), tid) for tid in str(matched).split(",") if tid
        )
    return pairs


def _build_candidate_keys(candidates: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            candidates["source1_entity_id"].astype(str),
            candidates["candidate_entity_id"].astype(str),
        )
    )


def _recall_stats(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    validation_s1_ids: set[str],
) -> dict:
    candidate_keys = _build_candidate_keys(candidates)
    present = missing = 0
    for s1_id, tid in _truth_pairs(ground_truth):
        if (s1_id, tid) in candidate_keys:
            present += 1
        else:
            missing += 1
    total = present + missing
    recall = present / total if total else 0.0

    per_s1 = (
        candidates.groupby("source1_entity_id").size()
        .reindex(list(validation_s1_ids), fill_value=0)
    )
    return {
        "total_ground_truth_matches": total,
        "candidate_present": present,
        "candidate_missing": missing,
        "candidate_recall": recall,
        "total_candidate_pairs": len(candidates),
        "average_candidates_per_s1": round(float(per_s1.mean()), 4),
        "median_candidates_per_s1":  round(float(per_s1.median()), 4),
        "maximum_candidates_per_s1": int(per_s1.max()),
    }


def _run_blocking(
    source1_val: pd.DataFrame,
    source2_full: pd.DataFrame,
    source3_full: pd.DataFrame,
    config: BlockingConfig,
    label: str,
) -> tuple[pd.DataFrame, dict, float, float]:
    """Run generate_candidates; return (candidates, stats, runtime_s, peak_mb)."""
    mode = blocking_mode(config)
    print(f"\n[Step 16] Running {label} blocking  (mode={mode}, "
          f"name_top_k={config.name_top_k}, address_top_k={config.address_top_k}) …")
    tracemalloc.start()
    t0 = time.perf_counter()
    candidates, _ = generate_candidates(source1_val, source2_full, source3_full, config)
    runtime = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 ** 2)
    print(f"[Step 16]   Done in {runtime:.1f}s  peak={peak_mb:.1f} MB  "
          f"pairs={len(candidates):,}")
    return candidates, {}, runtime, peak_mb


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_step16_experiment(
    experimental_name_top_k: int = EXPERIMENTAL_NAME_TOP_K,
    experimental_address_top_k: int = EXPERIMENTAL_ADDRESS_TOP_K,
    experimental_config: BlockingConfig | None = None,
) -> None:
    exp_config = experimental_config or BlockingConfig(
        name_top_k=experimental_name_top_k,
        address_top_k=experimental_address_top_k,
        approximate_target_rows=None,
        target_chunk_size=500_000,
        max_approx_candidates_per_query=500,
    )

    print(f"[Step 16] Baseline  : name_top_k={BASELINE_NAME_TOP_K}, "
          f"address_top_k={BASELINE_ADDRESS_TOP_K}")
    print(f"[Step 16] Experiment: name_top_k={exp_config.name_top_k}, "
          f"address_top_k={exp_config.address_top_k}")

    # ------------------------------------------------------------------
    # 1. Load shared data
    # ------------------------------------------------------------------
    print("[Step 16] Loading validation S1 IDs …")
    artifact = pd.read_csv(VALIDATION_PREDICTIONS_PATH, sep="\t", dtype=str)
    validation_s1_ids: set[str] = set(artifact["source1_entity_id"].astype(str).unique())
    print(f"[Step 16]   {len(validation_s1_ids):,} validation S1 entities")

    print("[Step 16] Loading training source files …")
    source1_full  = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    source2_full  = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    source3_full  = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str).fillna("")
    ground_truth_full = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str,
        usecols=["source1_entity_id", "matched_entity_ids"], keep_default_na=False,
    )
    source1_val = source1_full[
        source1_full["entity_id"].astype(str).isin(validation_s1_ids)
    ].reset_index(drop=True)
    ground_truth_val = ground_truth_full[
        ground_truth_full["source1_entity_id"].isin(validation_s1_ids)
    ].reset_index(drop=True)
    total_gt = _count_true_matches(ground_truth_val)
    print(f"[Step 16]   S1 val rows={len(source1_val):,}  GT matches={total_gt:,}")

    # ------------------------------------------------------------------
    # 2. Load baseline results (already computed in Step 14 — reuse them)
    # ------------------------------------------------------------------
    print("[Step 16] Loading Step 14 baseline results …")
    baseline_summary = pd.read_csv(BASELINE_SUMMARY_PATH, sep="\t")
    baseline_analysis = pd.read_csv(BASELINE_ANALYSIS_PATH, sep="\t", dtype=str)

    def _bv(metric: str):
        row = baseline_summary[baseline_summary["metric"] == metric]
        return row["value"].iloc[0] if not row.empty else None

    # The Step 14 summary uses different key names than the internal stats dict.
    # Map them explicitly; fall back to known Step 14 values if a key is absent.
    def _bv_float(key: str, fallback: float) -> float:
        v = _bv(key)
        return float(v) if v is not None else fallback

    baseline_stats = {
        "total_ground_truth_matches": int(_bv_float("total_ground_truth_matches", 3493)),
        "candidate_present":          int(_bv_float("candidate_present_matches", 3199)),
        "candidate_missing":          int(_bv_float("candidate_missing_matches", 294)),
        "candidate_recall":           _bv_float("candidate_generation_recall", 0.9158316633),
        # Step 14 summary does not store the raw pair count under a simple key;
        # derive it from average × validation count or fall back to the known value.
        "total_candidate_pairs":      int(
            round(_bv_float("average_candidates_per_source1", 44.262) * 1000)
        ),
        "average_candidates_per_s1":  _bv_float("average_candidates_per_source1", 44.262),
        "median_candidates_per_s1":   _bv_float("median_candidates_per_source1", 38.5),
        "maximum_candidates_per_s1":  int(_bv_float("maximum_candidates_per_source1", 396)),
        "runtime_seconds":            _bv_float("runtime_seconds", 4128.87),
        "peak_memory_mb":             _bv_float("peak_memory_mb", 5649.05),
    }

    print(f"[Step 16]   Baseline recall: {baseline_stats['candidate_recall']:.6f}  "
          f"present={baseline_stats['candidate_present']}  "
          f"pairs={baseline_stats['total_candidate_pairs']:,}")

    # ------------------------------------------------------------------
    # 3. Load the 294 misses and their reasons
    # ------------------------------------------------------------------
    misses_df = pd.read_csv(MISSES_ANALYSIS_PATH, sep="\t", dtype=str)
    miss_pairs: set[tuple[str, str]] = set(
        zip(misses_df["source1_entity_id"], misses_df["true_entity_id"])
    )
    miss_reason: dict[tuple[str, str], str] = {
        (str(r.source1_entity_id), str(r.true_entity_id)): str(r.candidate_missing_reason)
        for r in misses_df.itertuples()
    }
    print(f"[Step 16]   Loaded {len(miss_pairs)} previous candidate-missing pairs")

    # ------------------------------------------------------------------
    # 4. Run experimental blocking
    # ------------------------------------------------------------------
    exp_candidates, _, exp_runtime, exp_peak_mb = _run_blocking(
        source1_val, source2_full, source3_full, exp_config, "EXPERIMENTAL"
    )
    exp_stats = _recall_stats(exp_candidates, ground_truth_val, validation_s1_ids)
    exp_stats["runtime_seconds"] = round(exp_runtime, 2)
    exp_stats["peak_memory_mb"]  = round(exp_peak_mb, 2)

    print(f"\n[Step 16] Experimental recall: {exp_stats['candidate_recall']:.6f}  "
          f"present={exp_stats['candidate_present']}  "
          f"pairs={exp_stats['total_candidate_pairs']:,}")

    # ------------------------------------------------------------------
    # 5. Miss recovery analysis
    # ------------------------------------------------------------------
    exp_keys = _build_candidate_keys(exp_candidates)
    recovery_rows: list[dict] = []
    recovered_total    = 0
    recovered_by_cat: dict[str, int] = {}

    for _, miss_row in misses_df.iterrows():
        s1_id   = str(miss_row["source1_entity_id"])
        true_id = str(miss_row["true_entity_id"])
        prev_reason = str(miss_row["candidate_missing_reason"])
        key = (s1_id, true_id)
        is_recovered = key in exp_keys
        new_status   = "CANDIDATE_PRESENT" if is_recovered else "CANDIDATE_MISSING"
        new_reason   = "" if is_recovered else prev_reason
        if is_recovered:
            recovered_total += 1
            recovered_by_cat[prev_reason] = recovered_by_cat.get(prev_reason, 0) + 1

        recovery_rows.append({
            "source1_entity_id": s1_id,
            "true_entity_id":    true_id,
            "true_source":       str(miss_row["true_source"]),
            "previous_status":   "CANDIDATE_MISSING",
            "new_status":        new_status,
            "previous_reason":   prev_reason,
            "new_reason":        new_reason,
        })

    recovery_df = pd.DataFrame(recovery_rows, columns=[
        "source1_entity_id", "true_entity_id", "true_source",
        "previous_status", "new_status", "previous_reason", "new_reason",
    ])

    still_missing = int((recovery_df["new_status"] == "CANDIDATE_MISSING").sum())
    print(f"\n[Step 16] Recovered: {recovered_total} of {len(miss_pairs)} previous misses")
    print(f"[Step 16] Still missing: {still_missing}")
    print("[Step 16] Recovery by category:")
    for cat, cnt in sorted(recovered_by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:<50} +{cnt}")

    # ------------------------------------------------------------------
    # 6. Build comparison TSV
    # ------------------------------------------------------------------
    def _pct_change(base, exp_val) -> str:
        try:
            base_f = float(base)
            exp_f  = float(exp_val)
            if base_f == 0:
                return "N/A"
            return f"{(exp_f - base_f) / abs(base_f) * 100:+.4f}%"
        except (TypeError, ValueError):
            return "N/A"

    def _abs_change(base, exp_val) -> str:
        try:
            return f"{float(exp_val) - float(base):+.6g}"
        except (TypeError, ValueError):
            return "N/A"

    metrics = [
        ("baseline_name_top_k",             BASELINE_NAME_TOP_K,     exp_config.name_top_k),
        ("baseline_address_top_k",           BASELINE_ADDRESS_TOP_K,  exp_config.address_top_k),
        ("total_ground_truth_matches",       baseline_stats["total_ground_truth_matches"],
                                             exp_stats["total_ground_truth_matches"]),
        ("candidate_present",                baseline_stats["candidate_present"],
                                             exp_stats["candidate_present"]),
        ("candidate_missing",                baseline_stats["candidate_missing"],
                                             exp_stats["candidate_missing"]),
        ("candidate_recall",                 round(baseline_stats["candidate_recall"], 10),
                                             round(exp_stats["candidate_recall"], 10)),
        ("total_candidate_pairs",            baseline_stats["total_candidate_pairs"],
                                             exp_stats["total_candidate_pairs"]),
        ("average_candidates_per_s1",        baseline_stats["average_candidates_per_s1"],
                                             exp_stats["average_candidates_per_s1"]),
        ("median_candidates_per_s1",         baseline_stats["median_candidates_per_s1"],
                                             exp_stats["median_candidates_per_s1"]),
        ("maximum_candidates_per_s1",        baseline_stats["maximum_candidates_per_s1"],
                                             exp_stats["maximum_candidates_per_s1"]),
        ("candidate_set_growth_pct",         "0.0%",
                                             _pct_change(baseline_stats["total_candidate_pairs"],
                                                         exp_stats["total_candidate_pairs"])),
        ("recovered_from_294_misses",        0, recovered_total),
        ("still_missing",                    len(miss_pairs), still_missing),
        ("runtime_seconds",                  baseline_stats["runtime_seconds"],
                                             exp_stats["runtime_seconds"]),
        ("peak_memory_mb",                   baseline_stats["peak_memory_mb"],
                                             exp_stats["peak_memory_mb"]),
    ]

    comparison_rows = []
    for name, base_val, exp_val in metrics:
        comparison_rows.append({
            "metric":            name,
            "baseline":          base_val,
            "experimental":      exp_val,
            "absolute_change":   _abs_change(base_val, exp_val),
            "percentage_change": _pct_change(base_val, exp_val),
        })

    comparison_df = pd.DataFrame(comparison_rows, columns=[
        "metric", "baseline", "experimental", "absolute_change", "percentage_change"
    ])

    comparison_df.to_csv(COMPARISON_OUT, sep="\t", index=False)
    recovery_df.to_csv(RECOVERY_OUT,    sep="\t", index=False)
    print(f"\n[Step 16] Written: {COMPARISON_OUT}")
    print(f"[Step 16] Written: {RECOVERY_OUT}")

    # ------------------------------------------------------------------
    # 7. Console summary
    # ------------------------------------------------------------------
    recall_gain = exp_stats["candidate_recall"] - baseline_stats["candidate_recall"]
    growth_pct  = (exp_stats["total_candidate_pairs"] - baseline_stats["total_candidate_pairs"]
                   ) / baseline_stats["total_candidate_pairs"] * 100

    print("\n[Step 16] ══════════════════════════════════════════════════════")
    print(f"  Baseline  recall : {baseline_stats['candidate_recall']:.6f}"
          f"  ({baseline_stats['candidate_present']}/{baseline_stats['total_ground_truth_matches']})")
    print(f"  Exper.    recall : {exp_stats['candidate_recall']:.6f}"
          f"  ({exp_stats['candidate_present']}/{exp_stats['total_ground_truth_matches']})")
    print(f"  Recall gain      : {recall_gain:+.6f}")
    print(f"  Misses recovered : {recovered_total} / {len(miss_pairs)}")
    print(f"  Candidate growth : {growth_pct:+.2f}%  "
          f"({baseline_stats['total_candidate_pairs']:,} → {exp_stats['total_candidate_pairs']:,})")
    print(f"  Runtime          : {exp_stats['runtime_seconds']:.1f}s  "
          f"(baseline {baseline_stats['runtime_seconds']:.1f}s)")
    print(f"  Peak memory      : {exp_stats['peak_memory_mb']:.1f} MB  "
          f"(baseline {baseline_stats['peak_memory_mb']:.1f} MB)")
    print("[Step 16] ══════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    run_step16_experiment()
