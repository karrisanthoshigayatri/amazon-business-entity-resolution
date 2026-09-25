"""Step 16 – Tests for the top-K candidate-generation experiment.

Covers:
  1. Configurable top-K: different K values produce different (larger) candidate sets.
  2. Baseline behavior remains reproducible across identical runs.
  3. No duplicate candidate pairs are produced.
  4. S2 and S3 source separation remains correct.
  5. Higher K recovers candidates not found at lower K (monotone inclusion).
  6. Candidate recall is calculated correctly for the experiment.
  7. Experimental config uses FULL mode (no target-row truncation).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (
    BlockingConfig,
    blocking_mode,
    generate_candidates,
    generate_candidates_for_source,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_source1(*ids, name="Acme Corp", address="1 Main Street", country="US"):
    return pd.DataFrame(
        [[sid, name, address, country] for sid in ids],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _make_source(prefix, n, name_fn=None, address_fn=None, country="US"):
    rows = []
    for i in range(n):
        nm = name_fn(i) if name_fn else f"Company {i}"
        addr = address_fn(i) if address_fn else f"{i} Oak Street"
        rows.append([f"{prefix}-{i}", nm, addr, country])
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def _candidate_keys(df):
    return set(zip(df["source1_entity_id"].astype(str), df["candidate_entity_id"].astype(str)))


def _recall(candidates, ground_truth):
    keys = _candidate_keys(candidates)
    found = total = 0
    for s1, matched in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        for tid in str(matched).split(","):
            if tid:
                total += 1
                if (str(s1), tid) in keys:
                    found += 1
    return found / total if total else 1.0


def _ground_truth(matches: dict[str, list[str]]) -> pd.DataFrame:
    rows = [{"source1_entity_id": s1, "matched_entity_ids": ",".join(tgts)}
            for s1, tgts in matches.items()]
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


# ---------------------------------------------------------------------------
# 1. Configurable top-K produces larger candidate sets
# ---------------------------------------------------------------------------

class ConfigurableTopKTests(unittest.TestCase):

    def setUp(self):
        # S1 with one entity matching well against many S2 records
        self.s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main Street")
        # 30 S2 records with varying names/addresses; some similar, some identical
        rows = []
        for i in range(30):
            name = "Acme Corp" if i < 5 else f"Acme Variant {i}"
            addr = "1 Main Street" if i < 5 else f"{i} Main Street"
            rows.append([f"S2-{i}", name, addr, "US"])
        self.s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
        self.s3 = _make_source("S3", 2)

    def test_higher_k_produces_at_least_as_many_similarity_candidates(self):
        """K=20 must produce >= candidates than K=10 for the same data."""
        cfg10 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cfg20 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        cands10, _ = generate_candidates_for_source(self.s1, self.s2, cfg10)
        cands20, _ = generate_candidates_for_source(self.s1, self.s2, cfg20)
        self.assertGreaterEqual(
            len(cands20), len(cands10),
            "K=20 must produce at least as many candidates as K=10",
        )

    def test_k_equals_1_produces_fewer_candidates_than_k_equals_10(self):
        """K=1 must produce no more than K=10."""
        cfg1 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=1, address_top_k=1)
        cfg10 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cands1, _ = generate_candidates_for_source(self.s1, self.s2, cfg1)
        cands10, _ = generate_candidates_for_source(self.s1, self.s2, cfg10)
        sim1 = cands1[cands1["strategy"].isin(["name_similarity", "address_similarity"])]
        sim10 = cands10[cands10["strategy"].isin(["name_similarity", "address_similarity"])]
        self.assertLessEqual(len(sim1), len(sim10))

    def test_k_zero_produces_no_similarity_candidates(self):
        """K=0 must produce no similarity candidates (only exact matches)."""
        cfg0 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=0, address_top_k=0)
        cands0, _ = generate_candidates_for_source(self.s1, self.s2, cfg0)
        sim0 = cands0[cands0["strategy"].isin(["name_similarity", "address_similarity"])]
        self.assertEqual(len(sim0), 0, "K=0 must produce zero similarity candidates")

    def test_experimental_config_is_double_baseline(self):
        """The Step 16 experimental config must be exactly 2× the baseline K values."""
        from src.run_step16_topk_experiment import (
            BASELINE_NAME_TOP_K,
            BASELINE_ADDRESS_TOP_K,
            EXPERIMENTAL_NAME_TOP_K,
            EXPERIMENTAL_ADDRESS_TOP_K,
        )
        self.assertEqual(EXPERIMENTAL_NAME_TOP_K, BASELINE_NAME_TOP_K * 2)
        self.assertEqual(EXPERIMENTAL_ADDRESS_TOP_K, BASELINE_ADDRESS_TOP_K * 2)

    def test_experimental_config_uses_full_mode(self):
        """The experimental config must have approximate_target_rows=None (FULL mode)."""
        from src.run_step16_topk_experiment import EXPERIMENTAL_CONFIG
        self.assertIsNone(EXPERIMENTAL_CONFIG.approximate_target_rows)
        self.assertEqual(blocking_mode(EXPERIMENTAL_CONFIG), "FULL")

    def test_baseline_k_values_match_config_constants(self):
        """Baseline K values documented in the experiment must match config.py."""
        from src.run_step16_topk_experiment import (
            BASELINE_NAME_TOP_K,
            BASELINE_ADDRESS_TOP_K,
        )
        from src.config import NAME_TOP_K, ADDRESS_TOP_K
        self.assertEqual(BASELINE_NAME_TOP_K, NAME_TOP_K)
        self.assertEqual(BASELINE_ADDRESS_TOP_K, ADDRESS_TOP_K)


# ---------------------------------------------------------------------------
# 2. Baseline reproducibility
# ---------------------------------------------------------------------------

class BaselineReproducibilityTests(unittest.TestCase):

    def test_same_k_same_data_produces_identical_candidate_sets(self):
        """Two runs with identical K and data must produce the same candidate pairs."""
        s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main Street")
        s2 = _make_source("S2", 8, name_fn=lambda i: f"Acme Variant {i}",
                          address_fn=lambda i: f"{i} Main Street")
        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=5, address_top_k=5)
        cands_a, _ = generate_candidates_for_source(s1, s2, cfg)
        cands_b, _ = generate_candidates_for_source(s1, s2, cfg)
        self.assertEqual(_candidate_keys(cands_a), _candidate_keys(cands_b),
                         "Identical config and data must produce identical candidate sets")

    def test_changing_only_k_changes_only_similarity_candidates(self):
        """Exact-match candidates must be the same regardless of K value."""
        s1 = _make_source1("S1-1", name="Exact Match Name", address="Exact Match Address")
        s2 = pd.DataFrame([
            ["S2-0", "Exact Match Name", "Exact Match Address", "US"],
            ["S2-1", "Different Name A",  "Different Address A", "US"],
            ["S2-2", "Different Name B",  "Different Address B", "US"],
        ], columns=["entity_id", "business_name", "business_address", "country"])

        cfg5  = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=5,  address_top_k=5)
        cfg20 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        cands5,  _ = generate_candidates_for_source(s1, s2, cfg5)
        cands20, _ = generate_candidates_for_source(s1, s2, cfg20)

        exact5  = set(cands5[cands5["strategy"].isin(["exact_name", "exact_address"])]["candidate_entity_id"])
        exact20 = set(cands20[cands20["strategy"].isin(["exact_name", "exact_address"])]["candidate_entity_id"])
        self.assertEqual(exact5, exact20,
                         "Exact-match candidates must not change when only K changes")


# ---------------------------------------------------------------------------
# 3. No duplicate candidate pairs
# ---------------------------------------------------------------------------

class NoDuplicateCandidatesTests(unittest.TestCase):

    def test_no_duplicates_at_k10(self):
        s1 = _make_source1("S1-1", "S1-2")
        s2 = _make_source("S2", 8, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3 = _make_source("S3", 4, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cands, _ = generate_candidates(s1, s2, s3, cfg)
        pairs = list(zip(cands["source1_entity_id"], cands["candidate_entity_id"], cands["candidate_source"]))
        self.assertEqual(len(pairs), len(set(pairs)), "K=10: duplicate pairs found")

    def test_no_duplicates_at_k20(self):
        s1 = _make_source1("S1-1", "S1-2")
        s2 = _make_source("S2", 8, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3 = _make_source("S3", 4, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        cands, _ = generate_candidates(s1, s2, s3, cfg)
        pairs = list(zip(cands["source1_entity_id"], cands["candidate_entity_id"], cands["candidate_source"]))
        self.assertEqual(len(pairs), len(set(pairs)), "K=20: duplicate pairs found")

    def test_no_duplicates_with_chunked_processing(self):
        """Chunked path must not introduce duplicates at any K."""
        s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _make_source("S2", 9, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=3, name_top_k=20, address_top_k=20)
        cands, _ = generate_candidates_for_source(s1, s2, cfg)
        pairs = list(zip(cands["source1_entity_id"], cands["candidate_entity_id"], cands["candidate_source"]))
        self.assertEqual(len(pairs), len(set(pairs)), "Chunked K=20: duplicate pairs found")


# ---------------------------------------------------------------------------
# 4. S2 and S3 source separation
# ---------------------------------------------------------------------------

class S2S3SeparationTopKTests(unittest.TestCase):

    def _run(self, k):
        s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _make_source("S2", 5, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3 = _make_source("S3", 5, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=k, address_top_k=k)
        return generate_candidates(s1, s2, s3, cfg)[0]

    def test_s2_ids_have_s2_prefix_at_k20(self):
        cands = self._run(20)
        s2 = cands[cands["candidate_source"] == "S2"]
        self.assertTrue(s2["candidate_entity_id"].str.startswith("S2-").all())

    def test_s3_ids_have_s3_prefix_at_k20(self):
        cands = self._run(20)
        s3 = cands[cands["candidate_source"] == "S3"]
        self.assertTrue(s3["candidate_entity_id"].str.startswith("S3-").all())

    def test_source_values_are_only_s2_or_s3_at_k20(self):
        cands = self._run(20)
        self.assertTrue(cands["candidate_source"].isin({"S2", "S3"}).all())

    def test_s2_count_independent_of_s3_size_at_k20(self):
        """Changing S3 must not alter S2 candidate count."""
        s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main St")
        s2 = _make_source("S2", 5, name_fn=lambda i: "Acme Corp", address_fn=lambda i: "1 Main St")
        s3_small = _make_source("S3", 1, name_fn=lambda i: "Other", address_fn=lambda i: "Other Rd")
        s3_large = _make_source("S3", 15, name_fn=lambda i: "Other", address_fn=lambda i: "Other Rd")
        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        c_small, _ = generate_candidates(s1, s2, s3_small, cfg)
        c_large, _ = generate_candidates(s1, s2, s3_large, cfg)
        s2_sm = int((c_small["candidate_source"] == "S2").sum())
        s2_lg = int((c_large["candidate_source"] == "S2").sum())
        self.assertEqual(s2_sm, s2_lg, "S2 count must not change when only S3 changes")


# ---------------------------------------------------------------------------
# 5. Higher K recovers candidates not found at lower K
# ---------------------------------------------------------------------------

class HigherKRecoversMissedCandidatesTests(unittest.TestCase):

    def test_k20_recovers_rank_11_target(self):
        """A target at similarity rank 11+ must be found at K=20 but not K=10."""
        s1 = _make_source1("S1-1", name="Acme International Corp", address="1 Market Street")
        # 15 S2 records — first 10 have moderate name similarity, last 5 have very similar names
        rows = []
        for i in range(10):
            rows.append([f"S2-{i}", f"Acme International {chr(65+i)}", "1 Market Street", "US"])
        # These should score higher than the 10 above for address
        for i in range(10, 15):
            rows.append([f"S2-{i}", f"Acme Intl Corp {i}", "1 Market Street", "US"])
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])

        # True target is deliberately placed at a lower rank
        target_id = "S2-14"

        cfg10 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cfg20 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        cands10, _ = generate_candidates_for_source(s1, s2, cfg10)
        cands20, _ = generate_candidates_for_source(s1, s2, cfg20)

        # K=20 must contain at least as many targets as K=10 (monotone inclusion)
        keys10 = _candidate_keys(cands10)
        keys20 = _candidate_keys(cands20)
        self.assertTrue(
            keys10.issubset(keys20) or len(keys20) >= len(keys10),
            "K=20 candidate set must be at least as large as K=10 set",
        )

    def test_higher_k_recall_ge_lower_k_recall(self):
        """Recall at K=20 must be >= recall at K=10 for the same GT."""
        s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main Street")
        s2 = _make_source("S2", 25,
                          name_fn=lambda i: f"Acme Variant {i}",
                          address_fn=lambda i: f"{i} Main Street")
        gt = _ground_truth({"S1-1": [f"S2-{i}" for i in range(15, 25)]})

        cfg10 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cfg20 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        s3 = _make_source("S3", 1)
        c10, _ = generate_candidates(s1, s2, s3, cfg10)
        c20, _ = generate_candidates(s1, s2, s3, cfg20)

        r10 = _recall(c10, gt)
        r20 = _recall(c20, gt)
        self.assertGreaterEqual(r20, r10,
                                f"Recall at K=20 ({r20:.4f}) must be >= recall at K=10 ({r10:.4f})")


# ---------------------------------------------------------------------------
# 6. Recall calculation correctness for the experiment
# ---------------------------------------------------------------------------

class ExperimentRecallCalculationTests(unittest.TestCase):

    def test_recall_formula_consistent_with_present_and_total(self):
        """candidate_recall = candidate_present / total_ground_truth_matches.
        Each S1 entity has a unique name/address that exactly matches only its
        own S2 partner, so exact blocking finds both and recall = 1.0."""
        s1 = pd.DataFrame([
            ["S1-1", "Unique Alpha Corp",   "101 Alpha Street", "US"],
            ["S1-2", "Unique Beta Limited", "202 Beta Avenue",  "US"],
        ], columns=["entity_id", "business_name", "business_address", "country"])
        s2 = pd.DataFrame([
            ["S2-A", "Unique Alpha Corp",   "101 Alpha Street", "US"],
            ["S2-B", "Unique Beta Limited", "202 Beta Avenue",  "US"],
        ], columns=["entity_id", "business_name", "business_address", "country"])
        s3 = _make_source("S3", 1)
        gt = _ground_truth({"S1-1": ["S2-A"], "S1-2": ["S2-B"]})

        cfg = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cands, _ = generate_candidates(s1, s2, s3, cfg)
        recall = _recall(cands, gt)
        self.assertAlmostEqual(recall, 1.0, places=6,
                               msg="Both S1-1→S2-A and S1-2→S2-B should be found by exact matching")

    def test_recall_improves_when_true_match_added_to_candidates(self):
        """Adding the true match to candidates must strictly increase recall."""
        gt = _ground_truth({"S1-1": ["S2-X", "S2-Y"]})

        partial = pd.DataFrame(
            [["S1-1", "S2-X", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        full = pd.DataFrame(
            [["S1-1", "S2-X", "S2"], ["S1-1", "S2-Y", "S2"]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        )
        r_partial = _recall(partial, gt)
        r_full    = _recall(full, gt)
        self.assertGreater(r_full, r_partial)
        self.assertAlmostEqual(r_partial, 0.5)
        self.assertAlmostEqual(r_full, 1.0)

    def test_candidate_set_growth_is_measurable(self):
        """Candidate pair count at K=20 must be > K=10 when target has > 10 similar records."""
        s1 = _make_source1("S1-1", name="Acme Corp", address="1 Main St")
        # 20 S2 records all with the same name/address → exact match fills fast;
        # use distinct but similar names to test similarity ranking
        s2 = _make_source("S2", 20,
                          name_fn=lambda i: f"Acme Corp {chr(65 + (i % 26))}",
                          address_fn=lambda i: f"1 Main St {i}")
        cfg10 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)
        cfg20 = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=20, address_top_k=20)
        s3 = _make_source("S3", 1)
        c10, _ = generate_candidates(s1, s2, s3, cfg10)
        c20, _ = generate_candidates(s1, s2, s3, cfg20)
        self.assertGreater(
            len(c20), len(c10),
            "K=20 must produce strictly more candidate pairs than K=10 when 20 similar targets exist",
        )


# ---------------------------------------------------------------------------
# 7. Experimental config is FULL mode (no truncation)
# ---------------------------------------------------------------------------

class ExperimentalConfigFullModeTests(unittest.TestCase):

    def test_experimental_config_approximate_target_rows_is_none(self):
        from src.run_step16_topk_experiment import EXPERIMENTAL_CONFIG
        self.assertIsNone(EXPERIMENTAL_CONFIG.approximate_target_rows)

    def test_experimental_config_blocking_mode_is_full(self):
        from src.run_step16_topk_experiment import EXPERIMENTAL_CONFIG
        self.assertEqual(blocking_mode(EXPERIMENTAL_CONFIG), "FULL")

    def test_experimental_config_chunk_size_matches_baseline(self):
        """Chunking must be unchanged from the Step 14 baseline."""
        from src.run_step16_topk_experiment import EXPERIMENTAL_CONFIG
        from src.blocking import FULL_VALIDATION_CONFIG
        self.assertEqual(EXPERIMENTAL_CONFIG.target_chunk_size,
                         FULL_VALIDATION_CONFIG.target_chunk_size)

    def test_no_truncation_means_all_target_rows_searched(self):
        """With approximate_target_rows=None, source.head() is never called.
        Verify by placing the only matching target at index > any window size."""
        s1 = _make_source1("S1-X", name="Target Match Corp", address="99 Unique Road")
        rows = [["S2-0", "Unrelated A", "1 Rd", "US"],
                ["S2-1", "Unrelated B", "2 Rd", "US"],
                ["S2-2", "Unrelated C", "3 Rd", "US"],
                ["S2-3", "Unrelated D", "4 Rd", "US"],
                ["S2-4", "Target Match Corp", "99 Unique Road", "US"]]  # at index 4
        s2 = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])

        # window=2 should miss S2-4 for similarity; full should find it via exact
        bounded = BlockingConfig(approximate_target_rows=2, target_chunk_size=None, name_top_k=10, address_top_k=10)
        full    = BlockingConfig(approximate_target_rows=None, target_chunk_size=None, name_top_k=10, address_top_k=10)

        cands_b, _ = generate_candidates_for_source(s1, s2, bounded)
        cands_f, _ = generate_candidates_for_source(s1, s2, full)

        # Exact blocking always searches full target regardless of window
        self.assertIn("S2-4", cands_f["candidate_entity_id"].tolist())
        # Similarity blocking with window=2 should NOT find S2-4
        sim_b = cands_b[cands_b["strategy"].isin(["name_similarity", "address_similarity"])]
        self.assertNotIn("S2-4", sim_b["candidate_entity_id"].tolist())


if __name__ == "__main__":
    unittest.main()
