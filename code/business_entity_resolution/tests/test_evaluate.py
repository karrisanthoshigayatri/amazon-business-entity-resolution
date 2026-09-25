import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (
    evaluate_predictions,
    evaluate_persisted_validation_predictions,
    tune_persisted_thresholds,
)


class EvaluationTests(unittest.TestCase):
    def evaluate(self, predictions, truth, validation_ids=None):
        prediction_frame = pd.DataFrame(
            predictions,
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "match_probability"],
        )
        truth_frame = pd.DataFrame(
            truth,
            columns=["source1_entity_id", "matched_entity_ids"],
        )
        return evaluate_predictions(prediction_frame, truth_frame, 0.5, validation_ids)

    def test_correct_single_match(self):
        result = self.evaluate([["S1-1", "S2-1", "S2", 0.9]], [["S1-1", "S2-1"]])
        self.assertEqual(result["macro_f0_5"], 1.0)

    def test_correct_zero_match_prediction(self):
        result = self.evaluate([], [["S1-1", ""]], ["S1-1"])
        self.assertEqual(result["correctly_empty_count"], 1)
        self.assertEqual(result["macro_f0_5"], 1.0)

    def test_false_match_for_zero_match_entity(self):
        result = self.evaluate([["S1-1", "S2-1", "S2", 0.9]], [["S1-1", ""]])
        self.assertEqual(result["false_positive_count"], 1)
        self.assertEqual(result["macro_f0_5"], 0.0)

    def test_missed_single_match(self):
        result = self.evaluate([], [["S1-1", "S2-1"]], ["S1-1"])
        self.assertEqual(result["false_negative_count"], 1)
        self.assertEqual(result["macro_f0_5"], 0.0)

    def test_correct_multiple_matches(self):
        result = self.evaluate(
            [["S1-1", "S2-1", "S2", 0.9], ["S1-1", "S3-1", "S3", 0.8]],
            [["S1-1", "S2-1,S3-1"]],
        )
        self.assertEqual(result["macro_f0_5"], 1.0)

    def test_partial_multiple_match_and_extra_match(self):
        result = self.evaluate(
            [["S1-1", "S2-1", "S2", 0.9], ["S1-1", "S2-incorrect", "S2", 0.9]],
            [["S1-1", "S2-1,S3-1"]],
        )
        self.assertEqual(result["true_positive_count"], 1)
        self.assertEqual(result["false_positive_count"], 1)
        self.assertEqual(result["false_negative_count"], 1)

    def test_empty_prediction_and_ground_truth(self):
        result = self.evaluate([], [["S1-1", ""], ["S1-2", "S2-1"]], ["S1-1", "S1-2"])
        self.assertEqual(result["validation_entity_count"], 2)
        self.assertEqual(result["correctly_empty_count"], 1)

    def test_f05_formula(self):
        result = self.evaluate(
            [["S1-1", "S2-1", "S2", 0.9]],
            [["S1-1", "S2-1,S3-1"]],
        )
        expected = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        self.assertAlmostEqual(result["macro_f0_5"], expected)

    def test_macro_average(self):
        result = self.evaluate(
            [["S1-1", "S2-1", "S2", 0.9]],
            [["S1-1", "S2-1"], ["S1-2", ""]],
            ["S1-1", "S1-2"],
        )
        self.assertEqual(result["macro_f0_5"], 1.0)

    def test_persisted_predictions_use_threshold_and_labels(self):
        persisted = pd.DataFrame(
            [
                ["S1-1", "S2-1", "S2", 0.60, 1],
                ["S1-1", "S2-2", "S2", 0.40, 0],
                ["S1-2", "S3-1", "S3", 0.70, 0],
            ],
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "match_probability",
                "ground_truth_label",
            ],
        )
        result = evaluate_persisted_validation_predictions(persisted, threshold=0.50)
        self.assertEqual(result["true_positive_count"], 1)
        self.assertEqual(result["false_positive_count"], 1)
        self.assertEqual(result["false_negative_count"], 0)
        self.assertEqual(result["validation_entity_count"], 2)

    def test_multiple_thresholds_and_best_threshold(self):
        persisted = pd.DataFrame(
            [
                ["S1-1", "S2-1", "S2", 0.90, 1],
                ["S1-1", "S2-2", "S2", 0.20, 0],
                ["S1-2", "S2-3", "S2", 0.20, 0],
            ],
            columns=[
                "source1_entity_id", "candidate_entity_id", "candidate_source",
                "match_probability", "ground_truth_label",
            ],
        )
        comparison, tied = tune_persisted_thresholds(persisted, [0.2, 0.9])
        self.assertEqual(comparison.shape[0], 2)
        self.assertIn(0.9, tied)

    def test_tied_thresholds_are_all_returned(self):
        persisted = pd.DataFrame(
            [["S1-1", "S2-1", "S2", 0.9, 0]],
            columns=[
                "source1_entity_id", "candidate_entity_id", "candidate_source",
                "match_probability", "ground_truth_label",
            ],
        )
        _, tied = tune_persisted_thresholds(persisted, [0.1, 0.5, 0.9])
        self.assertEqual(tied, [0.1, 0.5, 0.9])


if __name__ == "__main__":
    unittest.main()
