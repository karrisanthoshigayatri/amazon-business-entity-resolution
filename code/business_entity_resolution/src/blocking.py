from __future__ import annotations

from typing import Iterable

from .normalization import normalize_name, normalize_address


def candidate_key(record: dict) -> tuple[str, str, str]:
    name = normalize_name(record.get("business_name", ""))
    address = normalize_address(record.get("business_address", ""))
    country = str(record.get("country", "")).strip().lower()
    return (country, name[:20], address[:20])


def build_candidate_pairs(source1_df, source2_df, source3_df, max_candidates_per_s1: int = 200):
    """Simple but extensible blocking baseline.

    It groups by country and name/address token prefixes to reduce comparison space.
    """
    source2_all = source2_df.copy()
    source3_all = source3_df.copy()

    all_records = pd.concat([source2_all, source3_all], ignore_index=True)

    candidates = {}
    for _, s1 in source1_df.iterrows():
        s1_id = s1["entity_id"]
        s1_name = normalize_name(s1.get("business_name", ""))
        s1_addr = normalize_address(s1.get("business_address", ""))
        s1_country = str(s1.get("country", "")).strip().lower()

        matches = []
        for _, rec in all_records.iterrows():
            rec_country = str(rec.get("country", "")).strip().lower()
            rec_name = normalize_name(rec.get("business_name", ""))
            rec_addr = normalize_address(rec.get("business_address", ""))

            if rec_country and s1_country and rec_country != s1_country:
                continue

            name_overlap = bool(set(s1_name.split()) & set(rec_name.split()))
            addr_overlap = bool(set(s1_addr.split()) & set(rec_addr.split()))
            if not (name_overlap or addr_overlap):
                continue

            matches.append(rec["entity_id"])

            if len(matches) >= max_candidates_per_s1:
                break

        candidates[s1_id] = matches

    return candidates
