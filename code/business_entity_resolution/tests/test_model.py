import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import split_by_source1_entity, train_baseline_model


class ModelTests(unittest.TestCase):
    def _features(self):
        rows = []
        for entity_number in range(12):
            for candidate_number in range(3):
                rows.append(
                    {
                        "source1_entity_id": f"S1-{entity_number}",
                        "candidate_entity_id": f"S2-{entity_number}-{candidate_number}",
                        "candidate_source": "S2" if candidate_number < 2 else "S3",
                        "name_exact_match": int(candidate_number == 0),
                        "name_token_jaccard": 1.0 if candidate_number == 0 else 0.1,
                        "address_exact_match": int(candidate_number == 0),
                        "country_exact_match": 1,
                        "same_source_indicator": int(candidate_number < 2),
                        "label": int(candidate_number == 0),
                    }
                )
        return pd.DataFrame(rows)

    def test_group_split_keeps_source1_entities_separate(self):
        train, validation = split_by_source1_entity(self._features(), validation_size=0.25)
        self.assertTrue(
            set(train.source1_entity_id).isdisjoint(set(validation.source1_entity_id))
        )

    def test_baseline_excludes_ids_and_outputs_probabilities(self):
        result = train_baseline_model(
            self._features(), n_estimators=10, max_depth=4, n_jobs=1
        )
        self.assertNotIn("source1_entity_id", result.feature_columns)
        self.assertNotIn("candidate_entity_id", result.feature_columns)
        self.assertNotIn("label", result.feature_columns)
        self.assertIn("match_probability", result.validation_features.columns)
        self.assertTrue(result.validation_features.match_probability.between(0, 1).all())
        self.assertEqual(result.diagnostics["class_weight"], "balanced_subsample")


if __name__ == "__main__":
    unittest.main()
