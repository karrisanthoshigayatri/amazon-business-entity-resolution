import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import assign_training_labels, build_pair_features


def records(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.s1 = records([
            ["S1-1", "Acme Co.", "12 Main St", "US"],
            ["S1-2", "", "", "France"],
        ])
        self.s2 = records([
            ["S2-1", "Acme Company", "12 Main Street", "US"],
            ["S2-2", "Different Shop", "99 Oak Road", "US"],
        ])
        self.s3 = records([
            ["S3-1", "Acme Co.", "12 Main St", "US"],
        ])

    def test_exact_name_address_country_and_numbers(self):
        candidates = pd.DataFrame([
            ["S1-1", "S2-1", "S2"],
        ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
        features = build_pair_features(candidates, self.s1, self.s2, self.s3)
        row = features.iloc[0]
        self.assertEqual(row.name_exact_match, 1)
        self.assertEqual(row.address_exact_match, 1)
        self.assertEqual(row.country_exact_match, 1)
        self.assertEqual(row.shared_digit_count, 2)
        self.assertEqual(row.name_token_jaccard, 1.0)
        self.assertEqual(row.address_token_jaccard, 1.0)

    def test_missing_values_and_source_indicator(self):
        candidates = pd.DataFrame([
            ["S1-2", "S3-1", "S3"],
        ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
        features = build_pair_features(candidates, self.s1, self.s2, self.s3)
        row = features.iloc[0]
        self.assertEqual(row.name_missing_s1, 1)
        self.assertEqual(row.address_missing_s1, 1)
        self.assertEqual(row.same_source_indicator, 0)

    def test_identical_high_and_different_low_similarity(self):
        candidates = pd.DataFrame([
            ["S1-1", "S3-1", "S3"],
            ["S1-1", "S2-2", "S2"],
        ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
        features = build_pair_features(candidates, self.s1, self.s2, self.s3)
        self.assertGreater(features.iloc[0].name_char_similarity, features.iloc[1].name_char_similarity)
        self.assertGreater(features.iloc[0].address_char_similarity, features.iloc[1].address_char_similarity)

    def test_label_generation(self):
        candidates = pd.DataFrame([
            ["S1-1", "S2-1", "S2"],
            ["S1-1", "S2-2", "S2"],
        ], columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
        ground_truth = pd.DataFrame([
            ["S1-1", "S2-1,S3-1"],
        ], columns=["source1_entity_id", "matched_entity_ids"])
        labeled = assign_training_labels(candidates, ground_truth)
        self.assertEqual(labeled.label.tolist(), [1, 0])

    def test_required_feature_columns_exist(self):
        candidates = pd.DataFrame([["S1-1", "S2-1", "S2"]], columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
        features = build_pair_features(candidates, self.s1, self.s2, self.s3)
        expected = {
            "name_exact_match", "name_token_jaccard", "name_char_similarity",
            "name_tfidf_similarity", "address_exact_match", "address_token_jaccard",
            "address_char_similarity", "country_exact_match", "name_missing_s1",
            "name_missing_target", "address_missing_s1", "address_missing_target",
            "same_source_indicator", "name_and_address_exact", "name_or_address_exact",
        }
        self.assertTrue(expected.issubset(features.columns))


if __name__ == "__main__":
    unittest.main()