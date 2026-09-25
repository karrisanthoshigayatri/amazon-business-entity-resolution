import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.error_analysis import build_error_summary, classify_error_rows


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame(
            [
                ["S1-1", "S2-1", "S2", 0.90, 1],
                ["S1-1", "S2-2", "S2", 0.80, 0],
                ["S1-2", "S2-3", "S2", 0.20, 1],
            ],
            columns=[
                "source1_entity_id", "candidate_entity_id", "candidate_source",
                "match_probability", "ground_truth_label",
            ],
        )
        self.features = pd.DataFrame(
            [
                ["S1-1", "S2-1", "S2", 1, .9, .9, .9, 1, .9, .9, 2, 0, 1, 1, 1, 2, 1, 0, 0],
                ["S1-1", "S2-2", "S2", 0, .1, .2, .1, 0, .2, .2, 0, 0, .1, .2, 1, 5, 0, 0, 0],
                ["S1-2", "S2-3", "S2", 0, .2, .2, .2, 0, .3, .3, 1, 0, .2, .2, 1, 4, 0, 0, 0],
            ],
            columns=[
                "source1_entity_id", "candidate_entity_id", "candidate_source",
                "name_exact_match", "name_token_jaccard", "name_char_similarity",
                "name_tfidf_similarity", "address_exact_match", "address_token_jaccard",
                "address_char_similarity", "shared_name_token_count", "shared_address_token_count",
                "shared_digit_count", "name_length_difference", "address_length_difference",
                "name_missing_s1", "name_missing_target", "address_missing_s1", "address_missing_target",
            ],
        )

    def test_classifies_false_positive_and_false_negative(self):
        errors = classify_error_rows(self.predictions, self.features, threshold=.45)
        self.assertIn("false_positive", set(errors.error_type))
        self.assertIn("false_negative", set(errors.error_type))
        self.assertIn("true_positive", set(errors.error_type))

    def test_threshold_filters_predictions(self):
        errors = classify_error_rows(self.predictions, self.features, threshold=.85)
        self.assertEqual(len(errors[errors.error_type == "true_positive"]), 1)
        self.assertEqual(len(errors[errors.error_type == "false_positive"]), 0)

    def test_summary_schema_and_counts(self):
        errors = classify_error_rows(self.predictions, self.features, threshold=.45)
        summary = build_error_summary(errors)
        self.assertEqual(summary.columns.tolist(), ["error_type", "count"])
        self.assertEqual(int(summary.loc[summary.error_type == "true_positive", "count"].iloc[0]), 1)
        self.assertEqual(int(summary.loc[summary.error_type == "false_positive", "count"].iloc[0]), 1)
        self.assertEqual(int(summary.loc[summary.error_type == "false_negative", "count"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()