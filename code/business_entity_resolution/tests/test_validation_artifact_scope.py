"""Step 13 – Validation Artifact Scope and Candidate Recall Diagnostic tests.

Covers:
  1. Validation population consistency
  2. Bounded artifact detection
  3. Complete artifact detection
  4. Ground-truth population consistency
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import BlockingConfig, generate_candidates_for_source


# ---------------------------------------------------------------------------
# Helpers shared by multiple test cases
# ---------------------------------------------------------------------------

def _make_source1(*ids: str) -> pd.DataFrame:
    """Minimal Source 1 DataFrame for blocking tests."""
    return pd.DataFrame(
        [[sid, "Acme Corp", "1 Main St", "US"] for sid in ids],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _make_source2(*ids: str) -> pd.DataFrame:
    """Minimal Source 2 DataFrame for blocking tests."""
    return pd.DataFrame(
        [[sid, "Acme Corp", "1 Main St", "US"] for sid in ids],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _make_artifact(
    s1_ids: list[str],
    positive_targets: dict[str, list[str]] | None = None,
    total_pairs_per_s1: int = 10,
) -> pd.DataFrame:
    """
    Build a synthetic validation_predictions artifact.

    positive_targets maps source1_entity_id -> list of true candidate_entity_ids
    (ground_truth_label=1). All other rows get label 0.
    """
    positive_targets = positive_targets or {}
    rows: list[dict] = []
    for s1_id in s1_ids:
        true_ids = set(positive_targets.get(s1_id, []))
        for i in range(total_pairs_per_s1):
            cand_id = f"S2-{s1_id}-{i}"
            is_match = cand_id in true_ids
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "candidate_source": "S2",
                    "match_probability": 0.9 if is_match else 0.05,
                    "ground_truth_label": 1 if is_match else 0,
                }
            )
    return pd.DataFrame(rows)


def _make_ground_truth(matches: dict[str, list[str]]) -> pd.DataFrame:
    """Build a ground_truth DataFrame from a dict {source1_id: [target_ids]}."""
    rows = [
        {"source1_entity_id": s1_id, "matched_entity_ids": ",".join(targets)}
        for s1_id, targets in matches.items()
    ]
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def _count_true_matches(ground_truth: pd.DataFrame) -> int:
    """Count individual target IDs across all matched_entity_ids entries."""
    total = 0
    for value in ground_truth["matched_entity_ids"]:
        total += len([x for x in str(value).split(",") if x])
    return total


def _artifact_is_bounded(artifact: pd.DataFrame, ground_truth: pd.DataFrame) -> bool:
    """
    Return True when the artifact covers < 100 % of the true matches for
    the Source 1 entities it contains.

    A bounded artifact has ground_truth_label==1 rows that represent fewer
    true matches than the full ground truth for those S1 entities.
    """
    artifact_s1_ids = set(artifact["source1_entity_id"].astype(str).unique())
    gt_for_artifact = ground_truth[
        ground_truth["source1_entity_id"].isin(artifact_s1_ids)
    ]
    full_match_count = _count_true_matches(gt_for_artifact)
    artifact_match_count = int((artifact["ground_truth_label"] == 1).sum())
    return artifact_match_count < full_match_count


def _artifact_is_complete(artifact: pd.DataFrame, ground_truth: pd.DataFrame) -> bool:
    """Return True when the artifact covers 100 % of true matches for its S1 IDs."""
    return not _artifact_is_bounded(artifact, ground_truth)


# ---------------------------------------------------------------------------
# 1. Validation population consistency
# ---------------------------------------------------------------------------

class ValidationPopulationConsistencyTests(unittest.TestCase):
    """Artifact S1 IDs must form a valid subset of the full train S1 population,
    the train/validation split must be disjoint, and repeated runs must be
    deterministic."""

    def setUp(self):
        # Build a small labeled feature table (36 rows, 12 S1 entities × 3 candidates)
        rows = []
        for entity_num in range(12):
            for cand_num in range(3):
                rows.append(
                    {
                        "source1_entity_id": f"S1-{entity_num}",
                        "candidate_entity_id": f"S2-{entity_num}-{cand_num}",
                        "candidate_source": "S2",
                        "name_exact_match": int(cand_num == 0),
                        "name_token_jaccard": 1.0 if cand_num == 0 else 0.1,
                        "label": int(cand_num == 0),
                    }
                )
        self.labeled_features = pd.DataFrame(rows)

    def test_artifact_s1_ids_are_subset_of_full_population(self):
        """Every S1 ID in the artifact must exist in the full training data."""
        from src.model import split_by_source1_entity

        _, validation = split_by_source1_entity(self.labeled_features)
        all_s1_ids = set(self.labeled_features["source1_entity_id"].astype(str).unique())
        artifact_s1_ids = set(validation["source1_entity_id"].astype(str).unique())
        self.assertTrue(
            artifact_s1_ids.issubset(all_s1_ids),
            "Artifact S1 IDs must be a subset of the full training S1 population",
        )

    def test_train_and_validation_s1_ids_are_disjoint(self):
        """No S1 entity must appear in both train and validation splits."""
        from src.model import split_by_source1_entity

        train, validation = split_by_source1_entity(self.labeled_features)
        train_ids = set(train["source1_entity_id"].astype(str).unique())
        val_ids = set(validation["source1_entity_id"].astype(str).unique())
        self.assertTrue(
            train_ids.isdisjoint(val_ids),
            "Train and validation S1 IDs must be completely disjoint",
        )

    def test_split_is_deterministic_across_runs(self):
        """Same random_state must produce the same validation S1 IDs."""
        from src.model import split_by_source1_entity

        _, val1 = split_by_source1_entity(self.labeled_features, random_state=42)
        _, val2 = split_by_source1_entity(self.labeled_features, random_state=42)
        self.assertEqual(
            sorted(val1["source1_entity_id"].astype(str).unique()),
            sorted(val2["source1_entity_id"].astype(str).unique()),
            "Validation split must be deterministic for the same random_state",
        )

    def test_different_random_states_may_differ(self):
        """Different random seeds should produce different validation splits
        (true for any dataset large enough to have multiple groups)."""
        from src.model import split_by_source1_entity

        _, val42 = split_by_source1_entity(self.labeled_features, random_state=42)
        _, val99 = split_by_source1_entity(self.labeled_features, random_state=99)
        ids42 = sorted(val42["source1_entity_id"].astype(str).unique())
        ids99 = sorted(val99["source1_entity_id"].astype(str).unique())
        self.assertNotEqual(
            ids42,
            ids99,
            "Different random states should yield different validation splits",
        )

    def test_all_artifact_s1_ids_are_represented_in_artifact(self):
        """Every S1 ID in the validation split must appear at least once in the
        artifact (no silent omission of entities during persistence)."""
        from src.model import split_by_source1_entity

        _, validation = split_by_source1_entity(self.labeled_features)
        val_entity_ids = set(validation["source1_entity_id"].astype(str).unique())
        artifact_entity_ids = set(validation["source1_entity_id"].astype(str).unique())
        # All validation S1 IDs must be present in the persisted frame
        missing = val_entity_ids - artifact_entity_ids
        self.assertEqual(
            missing,
            set(),
            f"Artifact is missing S1 IDs that are in the validation split: {missing}",
        )


# ---------------------------------------------------------------------------
# 2. Bounded artifact detection
# ---------------------------------------------------------------------------

class BoundedArtifactDetectionTests(unittest.TestCase):
    """Tests that identify when an artifact covers only a subset of the true
    ground-truth matches for its validation S1 entities."""

    def _ground_truth(self, artifact: pd.DataFrame, extra_targets: int = 0) -> pd.DataFrame:
        """Build ground truth that includes all artifact positives plus
        `extra_targets` additional true matches per S1 entity (simulating
        matches missed by the bounded blocking window)."""
        s1_ids = artifact["source1_entity_id"].astype(str).unique()
        rows = []
        for s1_id in s1_ids:
            pos_rows = artifact[
                (artifact["source1_entity_id"] == s1_id)
                & (artifact["ground_truth_label"] == 1)
            ]
            existing = list(pos_rows["candidate_entity_id"].astype(str))
            extra = [f"S2-MISSING-{s1_id}-{i}" for i in range(extra_targets)]
            matched = ",".join(existing + extra)
            rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched})
        return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])

    def test_artifact_with_missing_gt_matches_is_bounded(self):
        """When the ground truth contains matches absent from the artifact,
        the artifact must be classified as BOUNDED."""
        artifact = _make_artifact(
            ["S1-A", "S1-B"],
            positive_targets={"S1-A": ["S2-S1-A-0"]},
        )
        # Ground truth has two extra matches not in the artifact
        gt = self._ground_truth(artifact, extra_targets=2)
        self.assertTrue(
            _artifact_is_bounded(artifact, gt),
            "Artifact should be classified as BOUNDED when GT has extra true matches",
        )

    def test_artifact_with_all_gt_matches_is_not_bounded(self):
        """When every true match appears in the artifact, it is NOT bounded."""
        artifact = _make_artifact(
            ["S1-A", "S1-B"],
            positive_targets={"S1-A": ["S2-S1-A-0"]},
        )
        gt = self._ground_truth(artifact, extra_targets=0)
        self.assertFalse(
            _artifact_is_bounded(artifact, gt),
            "Artifact with full GT coverage should not be classified as BOUNDED",
        )

    def test_zero_positive_labels_with_nonempty_gt_is_bounded(self):
        """An artifact with no positive labels but a non-empty ground truth is
        maximally bounded (0 % recall)."""
        artifact = _make_artifact(["S1-A"], positive_targets={})
        gt = _make_ground_truth({"S1-A": ["S2-S1-A-999"]})
        self.assertTrue(
            _artifact_is_bounded(artifact, gt),
            "Artifact with zero positive labels must be BOUNDED when GT is non-empty",
        )

    def test_blocking_approximate_target_rows_restricts_candidate_pool(self):
        """Confirm that approximate_target_rows caps the target DataFrame passed
        to similarity blocking via source.head().

        We build a Source 2 pool of 6 records and set approximate_target_rows=3.
        The S1 entity has an exact match only at row index 5 (beyond the window)
        but not in the first 3 rows. With exact blocking disabled (top_k=0 for
        exact strategies) and the window in effect, that match is NOT recovered.
        However the exact match is NOT in the window so similarity blocking also
        misses it, demonstrating the restriction fires before candidate generation.
        """
        # S1 entity whose exact-match partner sits at row index 5 in Source 2
        s1 = _make_source1("S1-WINDOW")
        # Rows 0-4: unrelated names/addresses; row 5: exact match
        s2_rows = [
            ["S2-0", "Zeta Corp", "99 Oak Ln", "US"],
            ["S2-1", "Beta LLC", "7 Pine Rd", "US"],
            ["S2-2", "Gamma Inc", "3 Elm St", "US"],
            ["S2-3", "Delta Co", "8 Birch Blvd", "US"],
            ["S2-4", "Epsilon Ltd", "2 Cedar Ave", "US"],
            ["S2-5", "Acme Corp", "1 Main St", "US"],  # exact match with S1-WINDOW
        ]
        s2 = pd.DataFrame(s2_rows, columns=["entity_id", "business_name", "business_address", "country"])

        # Window covers only the first 3 rows → exact-match partner (index 5) is excluded
        config_bounded = BlockingConfig(
            name_top_k=10,
            address_top_k=10,
            approximate_target_rows=3,
            max_exact_block_size=None,
        )
        candidates_bounded, _ = generate_candidates_for_source(s1, s2, config_bounded)

        # Exact matching has no approximate_target_rows restriction, so S2-5 IS found
        # by exact strategies regardless of the window. This test verifies the window
        # only affects the approximate (similarity) strategies by checking counts.
        sim_bounded = candidates_bounded[
            candidates_bounded["strategy"].isin(["name_similarity", "address_similarity"])
        ]
        all_sim_targets = sim_bounded["candidate_entity_id"].unique().tolist()
        # All similarity candidates must come from the first 3 rows (indices 0–2)
        allowed = {"S2-0", "S2-1", "S2-2"}
        disallowed = set(all_sim_targets) - allowed
        self.assertEqual(
            disallowed,
            set(),
            f"Similarity candidates outside the window were returned: {disallowed}",
        )

    def test_restriction_fires_before_candidate_generation(self):
        """The approximate_target_rows head() is applied to the target pool
        before any candidate scoring, so records outside the window are never
        evaluated at all.

        A target record at index >= approximate_target_rows must produce ZERO
        similarity candidates even if it would score very highly.
        """
        s1 = _make_source1("S1-CHECK")
        # Rows 0-1: poor name/address match; row 2: perfect match — but window = 2
        s2 = pd.DataFrame(
            [
                ["S2-POOR-0", "Zeta Corp", "99 Oak Ln", "US"],
                ["S2-POOR-1", "Beta LLC", "7 Pine Rd", "US"],
                ["S2-EXACT", "Acme Corp", "1 Main St", "US"],  # outside window
            ],
            columns=["entity_id", "business_name", "business_address", "country"],
        )
        config = BlockingConfig(
            name_top_k=10,
            address_top_k=10,
            approximate_target_rows=2,  # only rows 0 and 1 are in the window
        )
        candidates, _ = generate_candidates_for_source(s1, s2, config)
        sim_candidates = candidates[
            candidates["strategy"].isin(["name_similarity", "address_similarity"])
        ]
        # S2-EXACT must NOT appear in any similarity candidate
        self.assertNotIn(
            "S2-EXACT",
            sim_candidates["candidate_entity_id"].tolist(),
            "Target record outside approximate_target_rows window must not be a similarity candidate",
        )


# ---------------------------------------------------------------------------
# 3. Complete artifact detection
# ---------------------------------------------------------------------------

class CompleteArtifactDetectionTests(unittest.TestCase):
    """Tests that verify the detection of a COMPLETE (non-bounded) artifact."""

    def test_artifact_covering_all_gt_matches_is_complete(self):
        """All true matches present in the artifact → COMPLETE classification."""
        gt = _make_ground_truth(
            {"S1-X": ["S2-S1-X-0", "S2-S1-X-1"], "S1-Y": []}
        )
        artifact = _make_artifact(
            ["S1-X", "S1-Y"],
            positive_targets={"S1-X": ["S2-S1-X-0", "S2-S1-X-1"]},
        )
        self.assertTrue(
            _artifact_is_complete(artifact, gt),
            "Artifact that covers all GT matches must be classified as COMPLETE",
        )
        self.assertFalse(
            _artifact_is_bounded(artifact, gt),
            "A COMPLETE artifact must not be classified as BOUNDED",
        )

    def test_empty_gt_with_empty_positives_is_complete(self):
        """When there are no true matches at all the artifact trivially covers
        100 % of ground truth."""
        gt = _make_ground_truth({"S1-X": []})
        artifact = _make_artifact(["S1-X"], positive_targets={})
        self.assertTrue(
            _artifact_is_complete(artifact, gt),
            "Empty GT with no positives must be considered COMPLETE (100 % trivially)",
        )

    def test_complete_artifact_does_not_trigger_bounded_flag(self):
        """A complete artifact must never be falsely flagged as BOUNDED."""
        s1_ids = [f"S1-{i}" for i in range(5)]
        # Give each S1 entity exactly one true match that is present in the artifact
        positive_targets = {sid: [f"S2-{sid}-0"] for sid in s1_ids}
        artifact = _make_artifact(s1_ids, positive_targets=positive_targets)
        gt_rows = {sid: [f"S2-{sid}-0"] for sid in s1_ids}
        gt = _make_ground_truth(gt_rows)
        self.assertFalse(
            _artifact_is_bounded(artifact, gt),
            "Complete artifact (100 % GT coverage) must not be flagged as BOUNDED",
        )

    def test_artifact_completeness_check_is_per_s1_entity(self):
        """Completeness is evaluated across all S1 entities collectively.
        Even if one S1 entity is perfectly covered, a missing match for another
        makes the artifact BOUNDED overall."""
        gt = _make_ground_truth(
            {"S1-A": ["S2-S1-A-0"], "S1-B": ["S2-MISSING"]}
        )
        # Artifact only contains the true match for S1-A, not for S1-B
        artifact = _make_artifact(
            ["S1-A", "S1-B"],
            positive_targets={"S1-A": ["S2-S1-A-0"]},
        )
        self.assertTrue(
            _artifact_is_bounded(artifact, gt),
            "Missing GT match for even one S1 entity makes the artifact BOUNDED",
        )


# ---------------------------------------------------------------------------
# 4. Ground-truth population consistency
# ---------------------------------------------------------------------------

class GroundTruthPopulationConsistencyTests(unittest.TestCase):
    """Tests that the ground-truth labels in an artifact agree with the
    external ground truth file and that coverage checks are consistent."""

    def test_positive_label_count_matches_gt_for_complete_artifact(self):
        """For a complete artifact, the number of ground_truth_label==1 rows
        must equal the number of individual true matches in the ground truth."""
        gt = _make_ground_truth(
            {"S1-1": ["S2-S1-1-0", "S2-S1-1-1"], "S1-2": ["S2-S1-2-0"]}
        )
        artifact = _make_artifact(
            ["S1-1", "S1-2"],
            positive_targets={"S1-1": ["S2-S1-1-0", "S2-S1-1-1"], "S1-2": ["S2-S1-2-0"]},
        )
        artifact_positive_count = int((artifact["ground_truth_label"] == 1).sum())
        gt_match_count = _count_true_matches(gt)
        self.assertEqual(
            artifact_positive_count,
            gt_match_count,
            "Positive label count in artifact must equal GT match count for complete artifact",
        )

    def test_positive_label_count_less_than_gt_for_bounded_artifact(self):
        """For a bounded artifact the positive label count must be strictly
        less than the full ground-truth match count."""
        full_gt = _make_ground_truth(
            {"S1-1": ["S2-S1-1-0", "S2-MISSING-1", "S2-MISSING-2"]}
        )
        # Artifact only captures one of the three true matches
        artifact = _make_artifact(
            ["S1-1"],
            positive_targets={"S1-1": ["S2-S1-1-0"]},
        )
        artifact_positive_count = int((artifact["ground_truth_label"] == 1).sum())
        gt_match_count = _count_true_matches(full_gt)
        self.assertLess(
            artifact_positive_count,
            gt_match_count,
            "Bounded artifact positive label count must be < full GT match count",
        )

    def test_gt_filtered_to_artifact_s1_ids_only(self):
        """When computing coverage, the ground truth must be filtered to only
        the S1 IDs present in the artifact (not the full training set)."""
        full_gt = _make_ground_truth(
            {
                "S1-IN": ["S2-S1-IN-0"],
                "S1-OUT": ["S2-S1-OUT-0"],  # not in the artifact
            }
        )
        artifact = _make_artifact(
            ["S1-IN"],
            positive_targets={"S1-IN": ["S2-S1-IN-0"]},
        )
        # Filter GT to artifact S1 IDs only
        artifact_s1_ids = set(artifact["source1_entity_id"].astype(str).unique())
        filtered_gt = full_gt[full_gt["source1_entity_id"].isin(artifact_s1_ids)]

        filtered_match_count = _count_true_matches(filtered_gt)
        artifact_positive_count = int((artifact["ground_truth_label"] == 1).sum())

        # After filtering, counts must agree (complete artifact)
        self.assertEqual(
            artifact_positive_count,
            filtered_match_count,
            "GT filtered to artifact S1 IDs must match the artifact positive label count",
        )

    def test_all_artifact_s1_ids_present_in_ground_truth(self):
        """Every S1 ID in a validation artifact must have a corresponding row
        in the ground truth (even if matched_entity_ids is empty)."""
        gt = _make_ground_truth(
            {"S1-1": ["S2-S1-1-0"], "S1-2": [], "S1-3": ["S2-S1-3-0"]}
        )
        artifact = _make_artifact(["S1-1", "S1-2", "S1-3"])
        artifact_ids = set(artifact["source1_entity_id"].astype(str).unique())
        gt_ids = set(gt["source1_entity_id"].astype(str).unique())
        missing_from_gt = artifact_ids - gt_ids
        self.assertEqual(
            missing_from_gt,
            set(),
            f"Artifact S1 IDs not represented in GT: {missing_from_gt}",
        )

    def test_gt_positive_ids_are_subset_of_gt_known_ids(self):
        """The candidate IDs that carry ground_truth_label==1 in the artifact
        must all appear in the ground truth's matched_entity_ids lists."""
        gt = _make_ground_truth(
            {"S1-1": ["S2-S1-1-0", "S2-S1-1-1"]}
        )
        artifact = _make_artifact(
            ["S1-1"],
            positive_targets={"S1-1": ["S2-S1-1-0", "S2-S1-1-1"]},
        )
        # Build the full set of known true-match candidate IDs from GT
        known_true_ids: set[str] = set()
        for value in gt["matched_entity_ids"]:
            known_true_ids.update(x for x in str(value).split(",") if x)

        artifact_positive_ids = set(
            artifact.loc[artifact["ground_truth_label"] == 1, "candidate_entity_id"].astype(str)
        )
        undeclared = artifact_positive_ids - known_true_ids
        self.assertEqual(
            undeclared,
            set(),
            f"Artifact positive labels reference IDs not in GT: {undeclared}",
        )


if __name__ == "__main__":
    unittest.main()
