"""Step 14 – Full-validation mode tests.

Covers:
  1. Full mode does not truncate the target DataFrame.
  2. Bounded mode still applies the configured limit when explicitly requested.
  3. Full mode includes records beyond row 250,000 when available.
  4. Candidate pairs remain deduplicated.
  5. Source 2 and Source 3 candidates are handled correctly.
  6. Candidate recall is calculated correctly.
  7. Zero-ground-truth-match handling works.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (
    BOUNDED_BLOCKING_CONFIG,
    FULL_VALIDATION_CONFIG,
    BlockingConfig,
    blocking_mode,
    combine_candidates,
    generate_candidates,
    generate_candidates_for_source,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _source1(*ids: str, name: str = "Acme Corp", address: str = "1 Main St", country: str = "US") -> pd.DataFrame:
    return pd.DataFrame(
        [[sid, name, address, country] for sid in ids],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _source(prefix: str, n: int, name_fn=None, address_fn=None) -> pd.DataFrame:
    """Build a source DataFrame with `n` rows prefixed by `prefix`."""
    rows = []
    for i in range(n):
        name = name_fn(i) if name_fn else f"Company {i}"
        address = address_fn(i) if address_fn else f"{i} Oak Street"
        rows.append([f"{prefix}-{i}", name, address, "US"])
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def _ground_truth(matches: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        {"source1_entity_id": s1_id, "matched_entity_ids": ",".join(targets)}
        for s1_id, targets in matches.items()
    ]
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def _count_true_matches(ground_truth: pd.DataFrame) -> int:
    return sum(
        len([x for x in str(v).split(",") if x])
        for v in ground_truth["matched_entity_ids"]
    )


def _recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Compute candidate_generation_recall against a ground_truth DataFrame."""
    candidate_keys = set(
        zip(candidates["source1_entity_id"].astype(str), candidates["candidate_entity_id"].astype(str))
    )
    found = total = 0
    for s1_id, matched in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        for tid in str(matched).split(","):
            if tid:
                total += 1
                if (str(s1_id), tid) in candidate_keys:
                    found += 1
    return found / total if total else 1.0


# ---------------------------------------------------------------------------
# 1. Full mode does not truncate the target DataFrame
# ---------------------------------------------------------------------------

class FullModeNoTruncationTests(unittest.TestCase):

    def test_full_config_has_none_approximate_target_rows(self):
        """FULL_VALIDATION_CONFIG must have approximate_target_rows=None."""
        self.assertIsNone(FULL_VALIDATION_CONFIG.approximate_target_rows)

    def test_blocking_mode_label_is_full(self):
        """blocking_mode() must return 'FULL' for FULL_VALIDATION_CONFIG."""
        self.assertEqual(blocking_mode(FULL_VALIDATION_CONFIG), "FULL")

    def test_full_mode_generates_candidates_from_entire_target(self):
        """With approximate_target_rows=None every target row is searched."""
        s1 = _source1("S1-1", name="Acme Corp", address="1 Main St")
        # Build source2 with 10 rows; exact match at each row
        s2 = _source("S2", 10, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        candidates, _ = generate_candidates_for_source(s1, s2, config)
        cand_ids = set(candidates["candidate_entity_id"].astype(str))
        # All 10 rows should appear (exact-name or exact-address hit)
        for i in range(10):
            self.assertIn(f"S2-{i}", cand_ids, f"S2-{i} missing from full-mode candidates")

    def test_full_mode_uses_chunked_path_when_target_exceeds_chunk_size(self):
        """When target > target_chunk_size, chunked path fires and still
        returns candidates from every chunk."""
        s1 = _source1("S1-A", name="Omega LLC", address="42 River Rd")
        # 9 rows, chunk_size=3 → 3 chunks; place exact match in last chunk
        rows = [
            ["S2-0", "Alpha Co", "1 Oak St", "US"],
            ["S2-1", "Beta Inc", "2 Pine Rd", "US"],
            ["S2-2", "Gamma Ltd", "3 Elm Ave", "US"],
            ["S2-3", "Delta Corp", "4 Birch Blvd", "US"],
            ["S2-4", "Epsilon LLC", "5 Cedar Ct", "US"],
            ["S2-5", "Zeta Co", "6 Maple Dr", "US"],
            ["S2-6", "Eta Inc", "7 Walnut Ln", "US"],
            ["S2-7", "Theta Ltd", "8 Chestnut Way", "US"],
            ["S2-8", "Omega LLC", "42 River Rd", "US"],   # exact match in chunk 3
        ]
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=3)
        candidates, _ = generate_candidates_for_source(s1, s2, config)
        cand_ids = set(candidates["candidate_entity_id"].astype(str))
        self.assertIn("S2-8", cand_ids, "Exact match in last chunk must be found in full chunked mode")


# ---------------------------------------------------------------------------
# 2. Bounded mode still applies the configured limit
# ---------------------------------------------------------------------------

class BoundedModeStillLimitsTests(unittest.TestCase):

    def test_bounded_config_has_250k_limit(self):
        """BOUNDED_BLOCKING_CONFIG must have approximate_target_rows=250_000."""
        self.assertEqual(BOUNDED_BLOCKING_CONFIG.approximate_target_rows, 250_000)

    def test_blocking_mode_label_is_bounded(self):
        """blocking_mode() must return 'BOUNDED(250000)' for BOUNDED_BLOCKING_CONFIG."""
        self.assertEqual(blocking_mode(BOUNDED_BLOCKING_CONFIG), "BOUNDED(250000)")

    def test_bounded_mode_excludes_records_beyond_window(self):
        """Similarity candidates for a record beyond the window must be absent
        in bounded mode but present in full mode."""
        s1 = _source1("S1-B", name="Unique Firm", address="99 Test Blvd")
        # 6 rows; only row 5 matches S1-B by name — beyond a window of 4
        rows = [
            ["S2-0", "Alpha Co", "1 Oak St", "US"],
            ["S2-1", "Beta Inc", "2 Pine Rd", "US"],
            ["S2-2", "Gamma Ltd", "3 Elm Ave", "US"],
            ["S2-3", "Delta Corp", "4 Birch Blvd", "US"],
            ["S2-4", "Epsilon LLC", "5 Cedar Ct", "US"],
            ["S2-5", "Unique Firm", "99 Test Blvd", "US"],  # beyond window=4
        ]
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])

        # Bounded window of 4 — S2-5 is beyond the similarity window
        bounded_cfg = BlockingConfig(approximate_target_rows=4, target_chunk_size=None)
        cands_bounded, _ = generate_candidates_for_source(s1, s2, bounded_cfg)
        sim_bounded = cands_bounded[cands_bounded["strategy"].isin(["name_similarity", "address_similarity"])]
        self.assertNotIn(
            "S2-5",
            sim_bounded["candidate_entity_id"].tolist(),
            "S2-5 must not appear in similarity candidates under bounded mode",
        )

        # Full mode — S2-5 should be found by exact blocking regardless
        full_cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        cands_full, _ = generate_candidates_for_source(s1, s2, full_cfg)
        self.assertIn(
            "S2-5",
            cands_full["candidate_entity_id"].tolist(),
            "S2-5 must appear in candidates under full mode (exact match)",
        )

    def test_custom_bounded_limit_respected(self):
        """A custom approximate_target_rows value must be respected exactly."""
        s1 = _source1("S1-C", name="Test Corp", address="10 Test Lane")
        s2 = _source("S2", 20,
                      name_fn=lambda i: "Test Corp" if i >= 10 else f"Other {i}",
                      address_fn=lambda i: "10 Test Lane" if i >= 10 else f"{i} Road")
        # Window of 5: all matches are at indices >= 10, so similarity misses them
        cfg = BlockingConfig(approximate_target_rows=5, target_chunk_size=None)
        cands, _ = generate_candidates_for_source(s1, s2, cfg)
        sim_cands = cands[cands["strategy"].isin(["name_similarity", "address_similarity"])]
        beyond_window = [cid for cid in sim_cands["candidate_entity_id"] if int(cid.split("-")[1]) >= 5]
        self.assertEqual(beyond_window, [], "No similarity candidate should come from beyond approximate_target_rows=5")


# ---------------------------------------------------------------------------
# 3. Full mode includes records beyond row 250,000
# ---------------------------------------------------------------------------

class FullModeIncludesBeyondWindowTests(unittest.TestCase):

    def test_full_mode_finds_exact_match_beyond_row_250k_position(self):
        """A record that would be at position > 250,000 in a real dataset must
        be reachable in full mode. We simulate this by setting a small window
        for bounded mode and confirming full mode lifts the restriction."""
        s1 = _source1("S1-D", name="FarCo", address="500 Far St")
        # Exact match lives at index 6 — beyond a window of 3
        rows = [
            ["S2-0", "NearA", "1 Nearby Rd", "US"],
            ["S2-1", "NearB", "2 Nearby Rd", "US"],
            ["S2-2", "NearC", "3 Nearby Rd", "US"],
            ["S2-3", "NearD", "4 Nearby Rd", "US"],
            ["S2-4", "NearE", "5 Nearby Rd", "US"],
            ["S2-5", "NearF", "6 Nearby Rd", "US"],
            ["S2-6", "FarCo", "500 Far St", "US"],  # exact match at index 6
        ]
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])

        # Bounded: window=3 → similarity misses S2-6; exact blocking finds it
        bounded = BlockingConfig(approximate_target_rows=3, target_chunk_size=None)
        cands_b, _ = generate_candidates_for_source(s1, s2, bounded)
        # Exact blocking always searches full target regardless of window,
        # so S2-6 should appear via exact_name / exact_address
        self.assertIn("S2-6", cands_b["candidate_entity_id"].tolist(),
                      "Exact match must be found by exact blocking even in bounded mode")

        # Full: everything found
        full = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        cands_f, _ = generate_candidates_for_source(s1, s2, full)
        self.assertIn("S2-6", cands_f["candidate_entity_id"].tolist(),
                      "Exact match must be found in full mode")

    def test_full_mode_similarity_reaches_beyond_simulated_250k_boundary(self):
        """Records with only similarity matches (not exact) beyond the bounded
        window must be found in full mode but not in bounded mode."""
        s1 = _source1("S1-E", name="Acme International", address="1 Global Plaza")
        # Index 5: similar but not exact name — beyond window=3
        rows = [
            ["S2-0", "Zeta Corp", "99 Oak Ln", "US"],
            ["S2-1", "Beta LLC", "7 Pine Rd", "US"],
            ["S2-2", "Gamma Inc", "3 Elm St", "US"],
            ["S2-3", "Delta Co", "8 Birch Blvd", "US"],
            ["S2-4", "Epsilon Ltd", "2 Cedar Ave", "US"],
            ["S2-5", "Acme Intl", "1 Global Plz", "US"],  # similar, beyond window=3
        ]
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])

        bounded = BlockingConfig(approximate_target_rows=3, target_chunk_size=None)
        cands_b, _ = generate_candidates_for_source(s1, s2, bounded)
        sim_b = cands_b[cands_b["strategy"].isin(["name_similarity", "address_similarity"])]
        self.assertNotIn("S2-5", sim_b["candidate_entity_id"].tolist(),
                         "S2-5 must not appear as similarity candidate in bounded mode (beyond window)")

        full = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        cands_f, _ = generate_candidates_for_source(s1, s2, full)
        sim_f = cands_f[cands_f["strategy"].isin(["name_similarity", "address_similarity"])]
        self.assertIn("S2-5", sim_f["candidate_entity_id"].tolist(),
                      "S2-5 must appear as similarity candidate in full mode")


# ---------------------------------------------------------------------------
# 4. Candidate pairs remain deduplicated
# ---------------------------------------------------------------------------

class CandidatePairDeduplicationTests(unittest.TestCase):

    def test_no_duplicate_pairs_in_full_mode(self):
        """Full mode must not produce duplicate (s1_id, candidate_id) pairs."""
        s1 = _source1("S1-1", "S1-2")
        s2 = _source("S2", 5, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        candidates, _ = generate_candidates_for_source(s1, s2, config)
        pairs = list(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"], candidates["candidate_source"]))
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate candidate pairs found in full mode")

    def test_no_duplicate_pairs_in_chunked_mode(self):
        """Chunked path must not introduce duplicates across chunk boundaries."""
        s1 = _source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _source("S2", 9, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        # chunk_size=3 → 3 chunks, same name everywhere → would duplicate without dedup
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=3)
        candidates, _ = generate_candidates_for_source(s1, s2, config)
        pairs = list(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"], candidates["candidate_source"]))
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate pairs found across chunk boundaries")

    def test_combine_candidates_deduplicates_s2_and_s3(self):
        """combine_candidates must deduplicate across S2 and S3 frames."""
        shared_row = pd.DataFrame(
            [["S1-1", "S2-1", "S2", "S2", "exact_name"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "source", "strategy"],
        )
        combined = combine_candidates([shared_row, shared_row])
        self.assertEqual(len(combined), 1, "Duplicate from two identical frames must be removed")


# ---------------------------------------------------------------------------
# 5. Source 2 and Source 3 candidates are handled correctly
# ---------------------------------------------------------------------------

class S2S3SeparationTests(unittest.TestCase):

    def test_s2_and_s3_candidates_have_correct_source_labels(self):
        """Candidates from S2 must be labelled S2; from S3 labelled S3."""
        s1 = _source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _source("S2", 3, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3 = _source("S3", 3, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        candidates, _ = generate_candidates(s1, s2, s3, config)
        s2_cands = candidates[candidates["candidate_source"] == "S2"]
        s3_cands = candidates[candidates["candidate_source"] == "S3"]
        self.assertTrue(s2_cands["candidate_entity_id"].str.startswith("S2-").all(),
                        "All S2-labelled candidates must have S2- prefixed IDs")
        self.assertTrue(s3_cands["candidate_entity_id"].str.startswith("S3-").all(),
                        "All S3-labelled candidates must have S3- prefixed IDs")

    def test_s2_and_s3_counts_are_independent(self):
        """Changing S3 contents must not affect S2 candidate count."""
        s1 = _source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _source("S2", 5, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3_small = _source("S3", 1, name_fn=lambda i: "Other", address_fn=lambda i: "Other Rd")
        s3_large = _source("S3", 10, name_fn=lambda i: "Other", address_fn=lambda i: "Other Rd")
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        cands_small, _ = generate_candidates(s1, s2, s3_small, config)
        cands_large, _ = generate_candidates(s1, s2, s3_large, config)
        s2_count_small = (cands_small["candidate_source"] == "S2").sum()
        s2_count_large = (cands_large["candidate_source"] == "S2").sum()
        self.assertEqual(s2_count_small, s2_count_large,
                         "S2 candidate count must not change when S3 contents change")

    def test_full_mode_produces_s2_and_s3_candidates(self):
        """Full mode must produce candidates from both S2 and S3."""
        s1 = _source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _source("S2", 3, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3 = _source("S3", 3, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        candidates, _ = generate_candidates(s1, s2, s3, FULL_VALIDATION_CONFIG)
        sources = set(candidates["candidate_source"].unique())
        self.assertIn("S2", sources, "Full mode must produce S2 candidates")
        self.assertIn("S3", sources, "Full mode must produce S3 candidates")


# ---------------------------------------------------------------------------
# 6. Candidate recall is calculated correctly
# ---------------------------------------------------------------------------

class CandidateRecallCalculationTests(unittest.TestCase):

    def test_perfect_recall_when_all_gt_present(self):
        """Recall must be 1.0 when every GT match is a candidate."""
        candidates = pd.DataFrame(
            [["S1-1", "S2-A", "S2"], ["S1-1", "S3-B", "S3"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        gt = _ground_truth({"S1-1": ["S2-A", "S3-B"]})
        self.assertAlmostEqual(_recall(candidates, gt), 1.0)

    def test_zero_recall_when_no_gt_present(self):
        """Recall must be 0.0 when no GT match is a candidate."""
        candidates = pd.DataFrame(
            [["S1-1", "S2-X", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        gt = _ground_truth({"S1-1": ["S2-A", "S3-B"]})
        self.assertAlmostEqual(_recall(candidates, gt), 0.0)

    def test_partial_recall(self):
        """Recall must be 0.5 when half the GT matches are candidates."""
        candidates = pd.DataFrame(
            [["S1-1", "S2-A", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        gt = _ground_truth({"S1-1": ["S2-A", "S3-B"]})
        self.assertAlmostEqual(_recall(candidates, gt), 0.5)

    def test_full_mode_achieves_higher_recall_than_bounded(self):
        """Full mode recall must be >= bounded mode recall on the same data."""
        s1 = _source1("S1-1", name="Unique Name", address="Unique Address")
        # Place the exact match at index 5 — beyond a window of 3
        rows = [
            ["S2-0", "Other A", "1 Rd", "US"],
            ["S2-1", "Other B", "2 Rd", "US"],
            ["S2-2", "Other C", "3 Rd", "US"],
            ["S2-3", "Other D", "4 Rd", "US"],
            ["S2-4", "Other E", "5 Rd", "US"],
            ["S2-5", "Unique Name", "Unique Address", "US"],
        ]
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
        gt = _ground_truth({"S1-1": ["S2-5"]})

        bounded_cfg = BlockingConfig(approximate_target_rows=3, target_chunk_size=None)
        full_cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        s3 = _source("S3", 1)

        cands_b, _ = generate_candidates(s1, s2, s3, bounded_cfg)
        cands_f, _ = generate_candidates(s1, s2, s3, full_cfg)

        recall_b = _recall(cands_b, gt)
        recall_f = _recall(cands_f, gt)
        self.assertGreaterEqual(recall_f, recall_b,
                                "Full mode recall must be >= bounded mode recall")
        self.assertAlmostEqual(recall_f, 1.0,
                               msg="Full mode must find the exact match and achieve recall=1.0")

    def test_recall_formula_matches_expected_value(self):
        """candidate_generation_recall = candidate_present / total_gt_matches."""
        candidates = pd.DataFrame(
            [["S1-1", "S2-A", "S2"], ["S1-2", "S2-C", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        gt = _ground_truth({"S1-1": ["S2-A", "S2-B"], "S1-2": ["S2-C", "S2-D"]})
        # 2 present out of 4 total
        self.assertAlmostEqual(_recall(candidates, gt), 0.5)


# ---------------------------------------------------------------------------
# 7. Zero-ground-truth-match handling
# ---------------------------------------------------------------------------

class ZeroGroundTruthMatchTests(unittest.TestCase):

    def test_recall_is_one_when_gt_is_empty(self):
        """When there are no GT matches recall is defined as 1.0 (nothing to miss)."""
        candidates = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"]
        )
        gt = _ground_truth({"S1-1": []})
        self.assertAlmostEqual(_recall(candidates, gt), 1.0)

    def test_recall_is_one_when_gt_has_empty_string_matches(self):
        """Empty matched_entity_ids string should produce zero true matches."""
        candidates = pd.DataFrame(
            [["S1-1", "S2-X", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        gt = pd.DataFrame(
            [["S1-1", ""]],
            columns=["source1_entity_id", "matched_entity_ids"],
        )
        self.assertAlmostEqual(_recall(candidates, gt), 1.0,
                               msg="Empty matched_entity_ids means zero GT matches → recall=1.0")

    def test_entities_with_no_gt_matches_do_not_affect_recall(self):
        """Entities with no GT matches must not reduce recall for those with matches."""
        candidates = pd.DataFrame(
            [["S1-1", "S2-A", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        gt = _ground_truth({"S1-1": ["S2-A"], "S1-2": []})
        # S1-2 has no GT matches; S1-1 has its match in candidates → recall=1.0
        self.assertAlmostEqual(_recall(candidates, gt), 1.0)

    def test_full_mode_candidates_generated_for_zero_match_entity(self):
        """Even an entity with no GT matches should still get candidates generated
        (blocking does not filter by GT)."""
        s1 = _source1("S1-NO-GT", name="Acme Corp", address="1 Main St")
        s2 = _source("S2", 3, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3 = _source("S3", 2)
        config = BlockingConfig(approximate_target_rows=None, target_chunk_size=None)
        candidates, _ = generate_candidates(s1, s2, s3, config)
        self.assertTrue(
            (candidates["source1_entity_id"] == "S1-NO-GT").any(),
            "Zero-GT entity must still have candidates generated in full mode",
        )


if __name__ == "__main__":
    unittest.main()
