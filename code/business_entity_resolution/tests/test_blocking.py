import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (
    BlockingConfig,
    combine_candidates,
    generate_candidates,
    generate_candidates_for_source,
)


def source1(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.s1 = source1(
            [
                ["S1-1", "Acme Co.", "1 Main St", "US"],
                ["S1-2", "", "", "US"],
            ]
        )
        self.s2 = source1(
            [
                ["S2-1", "Acme Company", "1 Main Street", "US"],
                ["S2-2", "Different Name", "9 Oak Road", "US"],
                ["S2-3", "Acme Company", "1 Main Street", "India"],
            ]
        )
        self.s3 = source1(
            [["S3-1", "Acme Co", "1 Main St", "US"]]
        )

    def test_exact_name_and_address_are_country_aware(self):
        candidates, stats = generate_candidates_for_source(
            self.s1,
            self.s2,
            BlockingConfig(name_top_k=0, address_top_k=0),
        )
        self.assertIn(("S1-1", "S2-1"), set(zip(candidates.source1_entity_id, candidates.candidate_entity_id)))
        self.assertNotIn(("S1-1", "S2-3"), set(zip(candidates.source1_entity_id, candidates.candidate_entity_id)))
        row = stats.loc[stats.source1_entity_id == "S1-1"].iloc[0]
        self.assertGreaterEqual(row.exact_name_count, 1)
        self.assertGreaterEqual(row.exact_address_count, 1)

    def test_missing_values_do_not_create_candidates(self):
        candidates, stats = generate_candidates_for_source(
            self.s1,
            self.s2,
            BlockingConfig(name_top_k=0, address_top_k=0),
        )
        self.assertEqual(stats.loc[stats.source1_entity_id == "S1-2", "final_candidate_count"].iloc[0], 0)
        self.assertFalse(candidates.source1_entity_id.eq("S1-2").any())

    def test_combination_deduplicates_pairs(self):
        frame = pd.DataFrame(
            [
                ["S1-1", "S2-1", "S2", "exact_name"],
                ["S1-1", "S2-1", "S2", "exact_address"],
            ],
            columns=["source1_entity_id", "candidate_entity_id", "source", "strategy"],
        )
        combined = combine_candidates([frame, frame])
        self.assertEqual(len(combined), 1)

    def test_source_separation_and_configurable_top_k(self):
        candidates, _ = generate_candidates(
            self.s1,
            self.s2,
            self.s3,
            BlockingConfig(name_top_k=1, address_top_k=1, approximate_max_target_rows=None),
        )
        self.assertTrue(set(candidates.source).issubset({"S2", "S3"}))
        self.assertFalse(candidates.candidate_entity_id.str.startswith("S1-").any())
        self.assertIn("S3-1", set(candidates.candidate_entity_id))

    def test_generation_does_not_create_cartesian_product(self):
        candidates, _ = generate_candidates_for_source(
            self.s1,
            self.s2,
            BlockingConfig(name_top_k=0, address_top_k=0),
        )
        self.assertLess(len(candidates), len(self.s1) * len(self.s2))


if __name__ == "__main__":
    unittest.main()