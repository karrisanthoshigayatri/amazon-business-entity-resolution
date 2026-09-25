import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
    preprocess_dataframe,
    preprocess_record,
)


class PreprocessingTests(unittest.TestCase):
    def test_name_normalization_handles_suffixes_and_punctuation(self):
        self.assertEqual(
            normalize_business_name("B+ Retail Inc"),
            "b retail incorporated",
        )
        self.assertEqual(
            normalize_business_name("Pvt. EFS Print Ventures Ltd."),
            "private efs print ventures limited",
        )
        self.assertEqual(normalize_business_name("Acme Co."), "acme company")

    def test_name_normalization_preserves_unicode_content(self):
        self.assertEqual(
            normalize_business_name("राम मार्केटिंग प्राइवेट लिमिटेड"),
            "राम मार्केटिंग प्राइवेट लिमिटेड",
        )

    def test_address_normalization_preserves_numbers_and_expands_abbreviations(self):
        self.assertEqual(
            normalize_business_address("105 ELM ST, MORGANTON, NC"),
            "105 elm street morganton nc",
        )
        self.assertEqual(
            normalize_business_address("Unit 11, 19 1/2 Stardust Trail"),
            "unit 11 19 1 2 stardust trail",
        )

    def test_missing_values_are_safe(self):
        self.assertEqual(normalize_business_name(None), "")
        self.assertEqual(normalize_business_address(float("nan")), "")
        self.assertEqual(normalize_country(pd.NA), "")

    def test_country_normalization_does_not_restrict_values(self):
        self.assertEqual(normalize_country("  France  "), "france")
        self.assertEqual(normalize_country(" South   Africa "), "south africa")

    def test_record_preserves_original_values_and_adds_derived_fields(self):
        record = preprocess_record(
            {
                "entity_id": "S1-1",
                "business_name": "Vision Partners Corp",
                "business_address": "IA, Iowa City, Unit 11",
                "country": " US ",
            }
        )
        self.assertEqual(record["business_name"], "Vision Partners Corp")
        self.assertEqual(record["business_name_normalized"], "vision partners corporation")
        self.assertEqual(record["name_tokens"], ("vision", "partners", "corporation"))
        self.assertEqual(record["name_token_count"], 3)
        self.assertTrue(record["has_name"])
        self.assertTrue(record["has_address"])
        self.assertEqual(record["country_normalized"], "us")

    def test_dataframe_preprocessing_preserves_rows_and_columns(self):
        data = pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme LLC",
                    "business_address": "1 Main St",
                    "country": "US",
                },
                {
                    "entity_id": "S1-2",
                    "business_name": "",
                    "business_address": "",
                    "country": "France",
                },
            ]
        )
        result = preprocess_dataframe(data)
        self.assertEqual(len(result), len(data))
        self.assertEqual(result["business_name"].tolist(), ["Acme LLC", ""])
        self.assertEqual(result["business_name_normalized"].tolist(), ["acme llc", ""])
        self.assertEqual(result["has_name"].tolist(), [True, False])
        self.assertEqual(result["has_address"].tolist(), [True, False])
        self.assertEqual(result["missing_name"].tolist(), [False, True])
        self.assertEqual(result["missing_address"].tolist(), [False, True])
        self.assertIn("name_tokens", result.columns)
        self.assertIn("address_tokens", result.columns)


if __name__ == "__main__":
    unittest.main()