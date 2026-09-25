from __future__ import annotations

import re
import unicodedata
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
    "co": "company",
    "company": "company",
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


def _replace_tokens(value: Any, replacements: dict[str, str]) -> str:
    return " ".join(
        replacements.get(token, token) for token in _normalize_characters(value).split()
    )


def normalize_text(value: Any) -> str:
    """Normalize Unicode, case, punctuation, and whitespace conservatively."""
    return _normalize_characters(value)


def normalize_business_name(value: Any) -> str:
    """Normalize a business name without removing meaningful words."""
    return _replace_tokens(value, LEGAL_SUFFIXES)


def normalize_business_address(value: Any) -> str:
    """Normalize address formatting and conservative common abbreviations."""
    return _replace_tokens(value, ADDRESS_ABBREVIATIONS)


def normalize_country(value: Any) -> str:
    """Normalize country formatting without restricting country values."""
    return normalize_text(value)


def _normalize_series(series: pd.Series, replacements: dict[str, str]) -> pd.Series:
    normalized = (
        series.astype("string")
        .fillna("")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    for source, target in replacements.items():
        normalized = normalized.str.replace(
            rf"(?<!\w){re.escape(source)}(?!\w)", target, regex=True
        )
    return normalized


def normalize_business_name_series(series: pd.Series) -> pd.Series:
    return _normalize_series(series, LEGAL_SUFFIXES)


def normalize_business_address_series(series: pd.Series) -> pd.Series:
    return _normalize_series(series, ADDRESS_ABBREVIATIONS)


def normalize_country_series(series: pd.Series) -> pd.Series:
    return _normalize_series(series, {})


# Backward-compatible names used by the existing blocking module.
normalize_name = normalize_business_name
normalize_address = normalize_business_address
