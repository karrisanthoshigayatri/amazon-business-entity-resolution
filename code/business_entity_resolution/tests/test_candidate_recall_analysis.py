import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate_recall_analysis import analyze_candidate_recall


class CandidateRecallTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame(
            [["S1-1", "S2-1", "S2", 0.8, 1]],
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "match_probability", "ground_truth_label"],
        )
        self.ground_truth = pd.DataFrame(
            [["S1-1", "S2-1,S3-1"]],
            columns=["source1_entity_id", "matched_entity_ids"],
        )
        self.features = pd.DataFrame(
            [["S1-1", "S2-1", "S2", 1, .9, .9, .9, 1, .9, .9, 1, 1, 2, 1, 1, 0, 1, 1, 1]],
            columns=[
                "source1_entity_id", "candidate_entity_id", "candidate_source",
                "name_exact_match", "name_token_jaccard", "name_char_similarity",
                "name_tfidf_similarity", "address_exact_match", "address_token_jaccard",
                "address_char_similarity", "shared_name_token_count", "shared_address_token_count",
                "shared_digit_count", "name_length_difference", "address_length_difference",
                "address_missing_target", "same_source_indicator", "country_exact_match", "name_or_address_exact",
            ],
        )

    def test_present_and_missing_detection(self):
        analysis, summary = analyze_candidate_recall(self.predictions, self.ground_truth, self.features)
        self.assertEqual(analysis.candidate_status.tolist(), ["CANDIDATE_PRESENT", "CANDIDATE_MISSING"])
        self.assertEqual(int(summary.loc[summary.metric == "candidate_present_matches", "value"].iloc[0]), 1)
        self.assertEqual(int(summary.loc[summary.metric == "candidate_missing_matches", "value"].iloc[0]), 1)

    def test_s2_s3_and_recall(self):
        analysis, summary = analyze_candidate_recall(self.predictions, self.ground_truth, self.features)
        self.assertEqual(set(analysis.true_source), {"S2", "S3"})
        recall = float(summary.loc[summary.metric == "candidate_generation_recall", "value"].iloc[0])
        self.assertEqual(recall, 0.5)

    def test_zero_ground_truth_matches(self):
        empty_truth = pd.DataFrame([["S1-1", ""]], columns=["source1_entity_id", "matched_entity_ids"])
        analysis, summary = analyze_candidate_recall(
            self.predictions.iloc[0:0], empty_truth, self.features.iloc[0:0]
        )
        self.assertTrue(analysis.empty)
        self.assertEqual(float(summary.loc[summary.metric == "candidate_generation_recall", "value"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()