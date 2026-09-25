from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .normalization import (
    normalize_business_address,
    normalize_business_address_series,
    normalize_business_name,
    normalize_business_name_series,
    normalize_country,
    normalize_country_series,
)


def preprocess_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one original record plus deterministic normalized fields."""
    original_name = record.get("business_name", "")
    original_address = record.get("business_address", "")
    normalized_name = normalize_business_name(original_name)
    normalized_address = normalize_business_address(original_address)
    name_tokens = tuple(normalized_name.split())
    address_tokens = tuple(normalized_address.split())

    enriched = dict(record)
    enriched.update(
        {
            "business_name_normalized": normalized_name,
            "business_address_normalized": normalized_address,
            "country_normalized": normalize_country(record.get("country", "")),
            "name_tokens": name_tokens,
            "address_tokens": address_tokens,
            "name_token_count": len(name_tokens),
            "address_token_count": len(address_tokens),
            "name_length": len(normalized_name),
            "address_length": len(normalized_address),
            "has_name": bool(normalized_name),
            "has_address": bool(normalized_address),
            "missing_name": not bool(normalized_name),
            "missing_address": not bool(normalized_address),
        }
    )
    return enriched


def preprocess_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Copy a source DataFrame and append normalized fields with vectorized pandas operations."""
    result = data.copy()
    names = result["business_name"] if "business_name" in result else pd.Series("", index=result.index)
    addresses = result["business_address"] if "business_address" in result else pd.Series("", index=result.index)
    countries = result["country"] if "country" in result else pd.Series("", index=result.index)

    result["business_name_normalized"] = normalize_business_name_series(names)
    result["business_address_normalized"] = normalize_business_address_series(addresses)
    result["country_normalized"] = normalize_country_series(countries)
    result["name_tokens"] = result["business_name_normalized"].str.split()
    result["address_tokens"] = result["business_address_normalized"].str.split()
    result["name_token_count"] = result["name_tokens"].str.len()
    result["address_token_count"] = result["address_tokens"].str.len()
    result["name_length"] = result["business_name_normalized"].str.len()
    result["address_length"] = result["business_address_normalized"].str.len()
    result["has_name"] = result["name_length"].gt(0)
    result["has_address"] = result["address_length"].gt(0)
    result["missing_name"] = ~result["has_name"]
    result["missing_address"] = ~result["has_address"]
    return result
