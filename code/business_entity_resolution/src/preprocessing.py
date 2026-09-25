from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import pandas as pd


LEGAL_SUFFIXES = {
    "ltd": "limited",
    "limited": "limited",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "llc": "llc",
    "pvt": "private",
    "private": "private",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "apt": "apartment",
    "apartment": "apartment",
    "no": "number",
    "number": "number",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _normalize_characters(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _safe_text(value)).casefold()
    characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isalnum() or category.startswith("M") or character.isspace():
            characters.append(character)
        else:
            characters.append(" ")
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def normalize_business_name(text: Any) -> str:
    """Normalize a business name while retaining meaningful tokens and suffixes."""
    tokens = _normalize_characters(text).split()
    return " ".join(LEGAL_SUFFIXES.get(token, token) for token in tokens)


def normalize_business_address(text: Any) -> str:
    """Normalize address formatting and conservative common abbreviations."""
    tokens = _normalize_characters(text).split()
    return " ".join(ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens)


def normalize_country(text: Any) -> str:
    """Normalize country casing and spacing without restricting country values."""
    return _normalize_characters(text)


def preprocess_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the original record plus deterministic normalized fields."""
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
        }
    )
    return enriched


def preprocess_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Copy a source DataFrame and append normalized and derived fields."""
    records = [preprocess_record(record) for record in data.to_dict("records")]
    return pd.DataFrame(records, index=data.index)