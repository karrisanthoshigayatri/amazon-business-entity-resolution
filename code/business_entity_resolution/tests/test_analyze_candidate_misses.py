"""Step 15 – Diagnostic tests for analyze_candidate_misses.py.

Tests cover:
  1. Similarity helper functions (char, token Jaccard, digit)
  2. Miss categorization logic for each major category
  3. Output schema validation (required columns present)
  4. Source 2 / Source 3 separation in the output
  5. Summary metric correctness
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyze_candidate_misses import (
    _categorize_miss,
    _char_sim,
    _digit_similarity,
    _token_jaccard,
    analyze_candidate_misses,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ANALYSIS_COLUMNS = [
    "source1_entity_id", "true_entity_id", "true_source",
    "normalized_name_s1", "normalized_name_true",
    "normalized_address_s1", "normalized_address_true",
    "country_s1", "country_true",
    "name_similarity", "address_similarity", "digit_similarity",
    "candidate_missing_reason", "failed_strategy",
]


def _make_full_analysis(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal full_analysis DataFrame accepted by analyze_candidate_misses."""
    return pd.DataFrame(rows, columns=["source1_entity_id", "true_entity_id", "true_source", "candidate_status"])


def _make_source_file(tmp_path: Path, filename: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, sep="\t", index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Similarity helper functions
# ---------------------------------------------------------------------------

class SimilarityHelperTests(unittest.TestCase):

    # --- _char_sim ---
    def test_char_sim_identical_strings(self):
        self.assertAlmostEqual(_char_sim("acme corp", "acme corp"), 1.0)

    def test_char_sim_empty_both(self):
        self.assertAlmostEqual(_char_sim("", ""), 1.0)

    def test_char_sim_one_empty(self):
        self.assertAlmostEqual(_char_sim("acme", ""), 0.0)
        self.assertAlmostEqual(_char_sim("", "acme"), 0.0)

    def test_char_sim_partial_overlap(self):
        sim = _char_sim("acme corp", "acme company")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_char_sim_completely_different(self):
        sim = _char_sim("aaaa", "zzzz")
        self.assertLess(sim, 0.5)

    # --- _token_jaccard ---
    def test_token_jaccard_identical(self):
        self.assertAlmostEqual(_token_jaccard("acme corp", "acme corp"), 1.0)

    def test_token_jaccard_no_overlap(self):
        self.assertAlmostEqual(_token_jaccard("alpha beta", "gamma delta"), 0.0)

    def test_token_jaccard_partial(self):
        j = _token_jaccard("acme corp llc", "acme inc")
        # shared={'acme'}, union={'acme','corp','llc','inc'} → 1/4 = 0.25
        self.assertAlmostEqual(j, 1 / 4)

    def test_token_jaccard_both_empty(self):
        self.assertAlmostEqual(_token_jaccard("", ""), 1.0)

    def test_token_jaccard_one_empty(self):
        self.assertAlmostEqual(_token_jaccard("acme", ""), 0.0)

    # --- _digit_similarity ---
    def test_digit_sim_identical_digits(self):
        self.assertAlmostEqual(_digit_similarity("123 main st", "123 main street"), 1.0)

    def test_digit_sim_no_digits(self):
        self.assertAlmostEqual(_digit_similarity("main street", "main road"), 1.0)

    def test_digit_sim_one_has_digits(self):
        self.assertAlmostEqual(_digit_similarity("123 main st", "main st"), 0.0)

    def test_digit_sim_different_digits(self):
        sim = _digit_similarity("123 oak st", "456 oak st")
        self.assertAlmostEqual(sim, 0.0)

    def test_digit_sim_partial_overlap(self):
        sim = _digit_similarity("123 456 oak st", "123 789 oak st")
        # shared={123}, union={123,456,789} → 1/3
        self.assertAlmostEqual(sim, 1 / 3)


# ---------------------------------------------------------------------------
# 2. Miss categorization logic
# ---------------------------------------------------------------------------

class CategorizeMissTests(unittest.TestCase):

    def _cat(self, ns1, ntgt, as1, atgt, cs1="us", ctgt="us"):
        from src.analyze_candidate_misses import _char_sim as cs
        reason, strategy = _categorize_miss(ns1, ntgt, as1, atgt, cs1, ctgt,
                                             cs(ns1, ntgt), cs(as1, atgt))
        return reason, strategy

    def test_country_mismatch(self):
        reason, strategy = self._cat("acme corp", "acme corp", "1 main st", "1 main st", "us", "india")
        self.assertEqual(reason, "country_mismatch")
        self.assertIn("exact_name_blocking", strategy)

    def test_missing_both_fields_s1(self):
        reason, _ = self._cat("", "", "", "1 main st")
        self.assertEqual(reason, "missing_both_fields_s1")

    def test_missing_both_fields_target(self):
        reason, _ = self._cat("acme corp", "", "1 main st", "")
        self.assertEqual(reason, "missing_both_fields_target")

    def test_missing_name_field_s1(self):
        reason, strategy = self._cat("", "acme corp", "1 main street", "1 main street")
        self.assertEqual(reason, "missing_name_field")
        self.assertIn("exact_name_blocking", strategy)

    def test_missing_address_field_target(self):
        reason, strategy = self._cat("acme corp", "acme corp", "1 main st", "")
        self.assertEqual(reason, "missing_address_field")
        self.assertIn("exact_address_blocking", strategy)

    def test_both_name_and_address_weak(self):
        # Very different name and address → both weak
        reason, strategy = self._cat("alpha corp", "zeta ltd", "1 oak road", "99 pine avenue")
        self.assertEqual(reason, "both_name_and_address_weak")
        self.assertIn("name_similarity", strategy)
        self.assertIn("address_similarity", strategy)

    def test_address_digit_variation(self):
        # Good name sim, but address digits differ
        reason, strategy = self._cat(
            "delhi chits private limited", "delhi chits limited",
            "je 2 block gali number 9 hari nagar", "dl je 02 west delhi"
        )
        self.assertEqual(reason, "address_digit_variation")
        self.assertIn("address_similarity", strategy)

    def test_near_match_outside_top_k(self):
        # Both name and address above 0.75 — pair was near-miss outside top-K
        reason, strategy = self._cat(
            "acme international corp", "acme international corporation",
            "4902 indiana avenue fort wayne", "4902 indiana avenue ft wayne indiana"
        )
        self.assertEqual(reason, "near_match_outside_top_k")
        self.assertIn("name_similarity", strategy)
        self.assertIn("address_similarity", strategy)

    def test_unexpected_miss_both_exact(self):
        # Exact name AND exact address — should be flagged as unexpected
        reason, _ = self._cat("acme corp", "acme corp", "1 main street", "1 main street")
        self.assertEqual(reason, "unexpected_miss_both_exact")

    def test_address_variation_near_threshold(self):
        # When name_sim >= 0.75 and addr_sim is in [_WEAK_SIM, 0.75) the
        # categoriser should return 'address_variation_near_threshold'.
        # We inject the similarity scores directly rather than deriving them
        # from raw strings, to avoid dependence on SequenceMatcher internals.
        # name_sim=0.80 (>=0.75), addr_sim=0.60 (in [0.5, 0.75))
        # addr token jaccard is also moderate so digit/token sub-branches don't fire
        reason, strategy = _categorize_miss(
            "sunrise bakery", "sunrise bakery inc",
            "1 main street north",   # addr token jaccard >= _WEAK_TOKEN
            "1 main street south",
            "us", "us",
            0.80,   # name_sim injected
            0.60,   # addr_sim injected
        )
        self.assertEqual(reason, "address_variation_near_threshold")
        self.assertIn("address_similarity", strategy)

    def test_name_variation_near_threshold(self):
        # When name_sim < 0.75 and addr_sim >= 0.75 the categoriser returns
        # a name-related reason.  We use name_sim=0.60 (< 0.75, >= _WEAK_SIM=0.5)
        # and addr_sim=0.85 (>=0.75), injected directly.
        reason, strategy = _categorize_miss(
            "acme trading", "beta trading",          # share 'trading' token
            "1 main street downtown", "1 main street downtown center",
            "us", "us",
            0.60,   # name_sim injected
            0.85,   # addr_sim injected
        )
        self.assertEqual(reason, "name_variation_near_threshold")
        self.assertIn("name_similarity", strategy)

    def test_reason_is_string(self):
        reason, strategy = self._cat("acme", "acme inc", "1 main st", "2 main road")
        self.assertIsInstance(reason, str)
        self.assertIsInstance(strategy, str)
        self.assertGreater(len(reason), 0)
        self.assertGreater(len(strategy), 0)


# ---------------------------------------------------------------------------
# 3. Output schema validation
# ---------------------------------------------------------------------------

class OutputSchemaTests(unittest.TestCase):

    def _run_analysis(self, tmp_path):
        """Run analyze_candidate_misses with minimal synthetic data."""
        # One CANDIDATE_PRESENT row + two CANDIDATE_MISSING rows
        full_analysis = _make_full_analysis([
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-B", "true_source": "S2", "candidate_status": "CANDIDATE_MISSING"},
            {"source1_entity_id": "S1-2", "true_entity_id": "S3-C", "true_source": "S3", "candidate_status": "CANDIDATE_MISSING"},
        ])
        analysis_path = tmp_path / "full_analysis.tsv"
        full_analysis.to_csv(analysis_path, sep="\t", index=False)

        _make_source_file(tmp_path, "train_source1.tsv", [
            {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Beta LLC", "business_address": "2 Oak Rd", "country": "US"},
        ])
        _make_source_file(tmp_path, "train_source2.tsv", [
            {"entity_id": "S2-B", "business_name": "Acme Inc", "business_address": "1 Main Street", "country": "US"},
        ])
        _make_source_file(tmp_path, "train_source3.tsv", [
            {"entity_id": "S3-C", "business_name": "Beta Limited", "business_address": "3 Pine Ave", "country": "US"},
        ])

        analysis_out = tmp_path / "misses_analysis.tsv"
        summary_out = tmp_path / "misses_summary.tsv"
        analysis_df, summary_df = analyze_candidate_misses(
            full_analysis_path=analysis_path,
            train_dir=tmp_path,
            misses_analysis_out=analysis_out,
            misses_summary_out=summary_out,
        )
        return analysis_df, summary_df, analysis_out, summary_out

    def test_analysis_columns_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            analysis_df, _, _, _ = self._run_analysis(Path(td))
        self.assertEqual(list(analysis_df.columns), ANALYSIS_COLUMNS)

    def test_analysis_row_count_equals_missing_count(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            analysis_df, _, _, _ = self._run_analysis(Path(td))
        # Two CANDIDATE_MISSING rows in the input
        self.assertEqual(len(analysis_df), 2)

    def test_summary_required_metrics_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, summary_df, _, _ = self._run_analysis(Path(td))
        required = {
            "total_true_matches", "candidate_present", "candidate_missing",
            "full_candidate_recall", "candidate_missing_rate",
            "candidate_missing_s2", "candidate_missing_s3",
        }
        summary_metrics = set(summary_df["metric"].astype(str))
        for m in required:
            self.assertIn(m, summary_metrics, f"Summary missing required metric: {m}")

    def test_output_files_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, _, analysis_out, summary_out = self._run_analysis(Path(td))
            self.assertTrue(analysis_out.exists(), "Analysis TSV was not written")
            self.assertTrue(summary_out.exists(), "Summary TSV was not written")

    def test_output_files_are_tab_separated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, _, analysis_out, summary_out = self._run_analysis(Path(td))
            df_a = pd.read_csv(analysis_out, sep="\t")
            df_s = pd.read_csv(summary_out, sep="\t")
        self.assertGreater(len(df_a.columns), 1)
        self.assertGreater(len(df_s.columns), 1)


# ---------------------------------------------------------------------------
# 4. Source 2 / Source 3 separation
# ---------------------------------------------------------------------------

class SourceSeparationTests(unittest.TestCase):

    def _run_mixed(self, tmp_path):
        full_analysis = _make_full_analysis([
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-X", "true_source": "S2", "candidate_status": "CANDIDATE_MISSING"},
            {"source1_entity_id": "S1-2", "true_entity_id": "S3-Y", "true_source": "S3", "candidate_status": "CANDIDATE_MISSING"},
        ])
        analysis_path = tmp_path / "full_analysis.tsv"
        full_analysis.to_csv(analysis_path, sep="\t", index=False)
        _make_source_file(tmp_path, "train_source1.tsv", [
            {"entity_id": "S1-1", "business_name": "Acme", "business_address": "1 St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Beta", "business_address": "2 Rd", "country": "US"},
        ])
        _make_source_file(tmp_path, "train_source2.tsv", [
            {"entity_id": "S2-X", "business_name": "Acme Inc", "business_address": "1 Street", "country": "US"},
        ])
        _make_source_file(tmp_path, "train_source3.tsv", [
            {"entity_id": "S3-Y", "business_name": "Beta Ltd", "business_address": "2 Road", "country": "US"},
        ])
        analysis_out = tmp_path / "a.tsv"
        summary_out = tmp_path / "s.tsv"
        return analyze_candidate_misses(analysis_path, tmp_path, analysis_out, summary_out)

    def test_true_source_values_are_s2_or_s3(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            df, _, = self._run_mixed(Path(td))
        self.assertTrue(df["true_source"].isin({"S2", "S3"}).all())

    def test_s2_and_s3_miss_counts_match_summary(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            df, summary, = self._run_mixed(Path(td))
        s2_in_analysis = int((df["true_source"] == "S2").sum())
        s3_in_analysis = int((df["true_source"] == "S3").sum())
        s2_in_summary = int(summary.loc[summary["metric"] == "candidate_missing_s2", "value"].iloc[0])
        s3_in_summary = int(summary.loc[summary["metric"] == "candidate_missing_s3", "value"].iloc[0])
        self.assertEqual(s2_in_analysis, s2_in_summary)
        self.assertEqual(s3_in_analysis, s3_in_summary)

    def test_s2_ids_start_with_s2_prefix(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            df, _ = self._run_mixed(Path(td))
        s2_rows = df[df["true_source"] == "S2"]
        self.assertTrue(s2_rows["true_entity_id"].str.startswith("S2-").all())

    def test_s3_ids_start_with_s3_prefix(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            df, _ = self._run_mixed(Path(td))
        s3_rows = df[df["true_source"] == "S3"]
        self.assertTrue(s3_rows["true_entity_id"].str.startswith("S3-").all())


# ---------------------------------------------------------------------------
# 5. Summary metric correctness
# ---------------------------------------------------------------------------

class SummaryMetricTests(unittest.TestCase):

    def _run(self, tmp_path, rows):
        full_analysis = _make_full_analysis(rows)
        analysis_path = tmp_path / "full_analysis.tsv"
        full_analysis.to_csv(analysis_path, sep="\t", index=False)
        _make_source_file(tmp_path, "train_source1.tsv", [
            {"entity_id": f"S1-{i}", "business_name": f"Corp {i}", "business_address": f"{i} Road", "country": "US"}
            for i in range(1, 5)
        ])
        _make_source_file(tmp_path, "train_source2.tsv", [
            {"entity_id": "S2-A", "business_name": "Corp A", "business_address": "1 Road", "country": "US"},
        ])
        _make_source_file(tmp_path, "train_source3.tsv", [
            {"entity_id": "S3-B", "business_name": "Corp B", "business_address": "2 Road", "country": "US"},
        ])
        return analyze_candidate_misses(analysis_path, tmp_path,
                                        tmp_path / "a.tsv", tmp_path / "s.tsv")

    def _get(self, summary_df, metric):
        row = summary_df[summary_df["metric"] == metric]
        self.assertFalse(row.empty, f"Metric '{metric}' not found in summary")
        return row["value"].iloc[0]

    def test_total_true_matches_counts_all_rows(self):
        import tempfile
        rows = [
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
            {"source1_entity_id": "S1-2", "true_entity_id": "S3-B", "true_source": "S3", "candidate_status": "CANDIDATE_MISSING"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, summary = self._run(Path(td), rows)
        self.assertEqual(int(self._get(summary, "total_true_matches")), 2)

    def test_candidate_present_and_missing_sum_to_total(self):
        import tempfile
        rows = [
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
            {"source1_entity_id": "S1-2", "true_entity_id": "S3-B", "true_source": "S3", "candidate_status": "CANDIDATE_MISSING"},
            {"source1_entity_id": "S1-3", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, summary = self._run(Path(td), rows)
        total = int(self._get(summary, "total_true_matches"))
        present = int(self._get(summary, "candidate_present"))
        missing = int(self._get(summary, "candidate_missing"))
        self.assertEqual(present + missing, total)

    def test_recall_equals_present_over_total(self):
        import tempfile
        rows = [
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
            {"source1_entity_id": "S1-2", "true_entity_id": "S3-B", "true_source": "S3", "candidate_status": "CANDIDATE_MISSING"},
            {"source1_entity_id": "S1-3", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
            {"source1_entity_id": "S1-4", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, summary = self._run(Path(td), rows)
        recall = float(self._get(summary, "full_candidate_recall"))
        self.assertAlmostEqual(recall, 3 / 4, places=6)

    def test_zero_missing_gives_zero_miss_rate(self):
        import tempfile
        rows = [
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_PRESENT"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, summary = self._run(Path(td), rows)
        missing = int(self._get(summary, "candidate_missing"))
        miss_rate = float(self._get(summary, "candidate_missing_rate"))
        self.assertEqual(missing, 0)
        self.assertAlmostEqual(miss_rate, 0.0)

    def test_miss_category_counts_present_when_misses_exist(self):
        import tempfile
        rows = [
            {"source1_entity_id": "S1-1", "true_entity_id": "S2-A", "true_source": "S2", "candidate_status": "CANDIDATE_MISSING"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, summary = self._run(Path(td), rows)
        cats = [m for m in summary["metric"].astype(str) if m.startswith("miss_category::")]
        self.assertGreater(len(cats), 0, "No miss_category:: entries found in summary")

    def test_failed_strategy_counts_present_when_misses_exist(self):
        import tempfile
        rows = [
            {"source1_entity_id": "S1-1", "true_entity_id": "S3-B", "true_source": "S3", "candidate_status": "CANDIDATE_MISSING"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, summary = self._run(Path(td), rows)
        strats = [m for m in summary["metric"].astype(str) if m.startswith("failed_strategy::")]
        self.assertGreater(len(strats), 0, "No failed_strategy:: entries found in summary")


if __name__ == "__main__":
    unittest.main()
