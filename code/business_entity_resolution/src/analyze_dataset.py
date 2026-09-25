from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_PATH = PROJECT_ROOT / "output" / "dataset_analysis.txt"
CHUNK_SIZE = 250_000

SOURCE_FILES = {
    "Source 1": TRAIN_DIR / "train_source1.tsv",
    "Source 2": TRAIN_DIR / "train_source2.tsv",
    "Source 3": TRAIN_DIR / "train_source3.tsv",
}
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"


def normalize_value(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def format_counter(values: Counter[str]) -> str:
    return ", ".join(f"{key or '[empty]'}: {value:,}" for key, value in values.items())


def format_number(value: float) -> str:
    return f"{value:,.2f}"


def update_length_stats(stats: dict[str, float], values: pd.Series) -> None:
    lengths = values.str.len()
    if lengths.empty:
        return
    stats["count"] += len(lengths)
    stats["sum"] += float(lengths.sum())
    stats["sum_squares"] += float((lengths * lengths).sum())
    stats["minimum"] = min(stats["minimum"], int(lengths.min()))
    stats["maximum"] = max(stats["maximum"], int(lengths.max()))


def finish_length_stats(stats: dict[str, float]) -> str:
    count = stats["count"]
    mean = stats["sum"] / count if count else 0
    variance = stats["sum_squares"] / count - mean * mean if count else 0
    return (
        f"min={int(stats['minimum']) if count else 0}, "
        f"max={int(stats['maximum']) if count else 0}, "
        f"mean={format_number(mean)}, "
        f"std={format_number(max(variance, 0) ** 0.5)}"
    )


def empty_length_stats() -> dict[str, float]:
    return {
        "count": 0,
        "sum": 0,
        "sum_squares": 0,
        "minimum": float("inf"),
        "maximum": 0,
    }


def collect_ground_truth() -> dict[str, object]:
    match_distribution: Counter[int] = Counter()
    source1_match_ids: dict[str, list[str]] = {}
    matched_ids: set[str] = set()
    source1_ids: set[str] = set()
    invalid_source1: list[str] = []
    invalid_matches: list[str] = []

    for chunk in pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        for source1_id, value in zip(
            chunk["source1_entity_id"], chunk["matched_entity_ids"]
        ):
            source1_ids.add(source1_id)
            if not source1_id.startswith("S1-") and len(invalid_source1) < 10:
                invalid_source1.append(source1_id)
            ids = [item for item in value.split(",") if item]
            match_distribution[len(ids)] += 1
            source1_match_ids[source1_id] = ids
            matched_ids.update(ids)
            for matched_id in ids:
                if not matched_id.startswith(("S2-", "S3-")) and len(invalid_matches) < 10:
                    invalid_matches.append(matched_id)

    return {
        "rows": sum(match_distribution.values()),
        "match_distribution": match_distribution,
        "source1_match_ids": source1_match_ids,
        "matched_ids": matched_ids,
        "source1_ids": source1_ids,
        "invalid_source1": invalid_source1,
        "invalid_matches": invalid_matches,
    }


def analyze_source(path: Path, label: str, target_ids: set[str]) -> dict[str, object]:
    rows = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    missing = Counter[str]()
    countries = Counter[str]()
    entity_ids: set[str] = set()
    duplicate_ids: Counter[str] = Counter()
    raw_names: set[str] = set()
    normalized_names: set[str] = set()
    raw_addresses: set[str] = set()
    normalized_addresses: set[str] = set()
    name_lengths = empty_length_stats()
    address_lengths = empty_length_stats()
    invalid_prefix: list[str] = []
    expected_prefix = f"S{label[-1]}-"
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        if not columns:
            columns = chunk.columns.tolist()
            dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}
        rows += len(chunk)
        missing.update(chunk.eq("").sum().to_dict())
        countries.update(chunk["country"].tolist())
        entity_ids.update(chunk["entity_id"].tolist())
        duplicate_ids.update(chunk["entity_id"].value_counts().to_dict())

        names = chunk["business_name"]
        addresses = chunk["business_address"]
        raw_names.update(names.tolist())
        raw_addresses.update(addresses.tolist())
        normalized_names.update(names.map(normalize_value).tolist())
        normalized_addresses.update(addresses.map(normalize_value).tolist())
        update_length_stats(name_lengths, names)
        update_length_stats(address_lengths, addresses)

        bad_ids = chunk.loc[
            ~chunk["entity_id"].str.startswith(expected_prefix), "entity_id"
        ]
        invalid_prefix.extend(bad_ids.head(max(0, 10 - len(invalid_prefix))).tolist())

        if target_ids and len(examples["matched_records"]) < 20:
            matched = chunk[chunk["entity_id"].isin(target_ids)]
            for record in matched.to_dict("records"):
                examples["matched_records"].append(record)

    duplicates = {key: count for key, count in duplicate_ids.items() if count > 1}
    return {
        "label": label,
        "rows": rows,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing,
        "countries": countries,
        "entity_ids": entity_ids,
        "duplicates": duplicates,
        "raw_names": raw_names,
        "normalized_names": normalized_names,
        "raw_addresses": raw_addresses,
        "normalized_addresses": normalized_addresses,
        "name_lengths": name_lengths,
        "address_lengths": address_lengths,
        "invalid_prefix": invalid_prefix,
        "examples": examples,
    }


def make_noise_examples(
    source_results: dict[str, dict[str, object]],
    ground_truth: dict[str, object],
) -> dict[str, list[str]]:
    examples = {
        "name differences": [],
        "address differences": [],
        "legal suffix or punctuation differences": [],
        "address formatting differences": [],
    }
    records_by_id: dict[str, dict[str, str]] = {}
    for result in source_results.values():
        for record in result["examples"]["matched_records"]:
            records_by_id[record["entity_id"]] = record

    for source1_id, matched_ids in ground_truth["source1_match_ids"].items():
        source1 = records_by_id.get(source1_id)
        if not source1:
            continue
        for matched_id in matched_ids:
            matched = records_by_id.get(matched_id)
            if not matched:
                continue
            name_differs = normalize_value(source1["business_name"]) != normalize_value(
                matched["business_name"]
            )
            address_differs = normalize_value(
                source1["business_address"]
            ) != normalize_value(matched["business_address"])
            if name_differs and len(examples["name differences"]) < 3:
                examples["name differences"].append(
                    f"confirmed pair {source1_id} / {matched_id}: "
                    f"{source1['business_name']!r} -> {matched['business_name']!r}"
                )
            if address_differs and len(examples["address differences"]) < 3:
                examples["address differences"].append(
                    f"confirmed pair {source1_id} / {matched_id}: "
                    f"{source1['business_address']!r} -> {matched['business_address']!r}"
                )
            if name_differs and re.search(
                r"\b(ltd|limited|inc|corp|llc|llp|pvt|private)\b|[^\w\s]",
                source1["business_name"].lower() + " " + matched["business_name"].lower(),
            ) and len(examples["legal suffix or punctuation differences"]) < 3:
                examples["legal suffix or punctuation differences"].append(
                    f"confirmed pair {source1_id} / {matched_id}: "
                    f"{source1['business_name']!r} -> {matched['business_name']!r}"
                )
            if address_differs and re.search(
                r"\b(st|street|rd|road|ave|avenue|blvd|boulevard)\b|[,.-]",
                source1["business_address"].lower() + " " + matched["business_address"].lower(),
            ) and len(examples["address formatting differences"]) < 3:
                examples["address formatting differences"].append(
                    f"confirmed pair {source1_id} / {matched_id}: "
                    f"{source1['business_address']!r} -> {matched['business_address']!r}"
                )
        if all(len(values) >= 3 for values in examples.values()):
            break
    return examples


def write_report(ground_truth: dict[str, object], results: dict[str, dict[str, object]]) -> str:
    lines = ["DATASET ANALYSIS", "=" * 72, "TRAINING DATASET ONLY", ""]
    lines.append("RECORD COUNTS")
    for label, result in results.items():
        lines.append(f"{label}: {result['rows']:,}")
    lines.append(f"Ground Truth: {ground_truth['rows']:,}\n")

    lines.append("SCHEMA, TYPES, AND MISSING VALUES")
    for label, result in results.items():
        lines.extend(
            [
                f"{label} columns: {result['columns']}",
                f"{label} data types: {result['dtypes']}",
                f"{label} missing values: {format_counter(result['missing'])}",
            ]
        )
    lines.append("")

    lines.append("BUSINESS NAME AND ADDRESS ANALYSIS")
    for label, result in results.items():
        lines.extend(
            [
                f"{label} missing names: {result['missing']['business_name']:,}",
                f"{label} unique raw names: {len(result['raw_names']):,}",
                f"{label} duplicate raw names: {result['rows'] - len(result['raw_names']):,}",
                f"{label} unique normalized names: {len(result['normalized_names']):,}",
                f"{label} missing addresses: {result['missing']['business_address']:,}",
                f"{label} unique raw addresses: {len(result['raw_addresses']):,}",
                f"{label} duplicate raw addresses: {result['rows'] - len(result['raw_addresses']):,}",
                f"{label} name length statistics: {finish_length_stats(result['name_lengths'])}",
                f"{label} address length statistics: {finish_length_stats(result['address_lengths'])}",
            ]
        )
    lines.append("")

    lines.append("COUNTRY ANALYSIS")
    country_sets = {label: set(result["countries"]) for label, result in results.items()}
    for label, result in results.items():
        lines.append(f"{label}: {len(result['countries'])} countries; {format_counter(result['countries'])}")
    shared = set.intersection(*country_sets.values())
    lines.append(f"Countries shared across all three sources: {sorted(shared)}")
    for label, countries in country_sets.items():
        only_here = countries - set.union(*(other for other_label, other in country_sets.items() if other_label != label))
        lines.append(f"Countries only in {label}: {sorted(only_here)}")
    lines.append("")

    lines.append("ENTITY ID ANALYSIS")
    for label, result in results.items():
        lines.append(
            f"{label}: duplicate entity IDs={len(result['duplicates'])}; "
            f"invalid prefix examples={result['invalid_prefix']}"
        )
    lines.append("")

    distribution = ground_truth["match_distribution"]
    lines.append("GROUND-TRUTH ANALYSIS")
    lines.append(f"Match distribution: {dict(sorted(distribution.items()))}")
    lines.append(f"Zero matches: {distribution[0]:,}")
    lines.append(f"Exactly one match: {distribution[1]:,}")
    lines.append(f"Multiple matches: {sum(count for number, count in distribution.items() if number > 1):,}")
    lines.append(f"Invalid Source 1 ID examples: {ground_truth['invalid_source1']}")
    lines.append(f"Invalid matched ID examples: {ground_truth['invalid_matches']}")
    lines.append("")

    lines.append("MATCHING COVERAGE")
    coverage = {"at least one S2": 0, "at least one S3": 0, "both S2 and S3": 0, "no matches": 0}
    for matched_ids in ground_truth["source1_match_ids"].values():
        has_s2 = any(item.startswith("S2-") for item in matched_ids)
        has_s3 = any(item.startswith("S3-") for item in matched_ids)
        if not matched_ids:
            coverage["no matches"] += 1
        if has_s2:
            coverage["at least one S2"] += 1
        if has_s3:
            coverage["at least one S3"] += 1
        if has_s2 and has_s3:
            coverage["both S2 and S3"] += 1
    lines.extend(f"{key}: {value:,}" for key, value in coverage.items())
    lines.append("")

    lines.append("GROUND-TRUTH-CONFIRMED NOISE EXAMPLES")
    noise_examples = make_noise_examples(results, ground_truth)
    lines.append("Examples are confirmed linked pairs; this report does not independently decide matches.")
    for category, examples in noise_examples.items():
        lines.append(f"{category}:")
        lines.extend(f"- {example}" for example in examples or ["No example collected"])
    lines.append("")

    lines.append("CANDIDATE-GENERATION CONSIDERATIONS")
    lines.extend(
        [
            "Useful blocking fields: country, business_name, business_address, and their normalized tokens.",
            "High-missingness field: business_address is missing in Source 2 and Source 3; business_name has no observed missing values.",
            "Normalization needs: case, punctuation, whitespace, legal suffixes, abbreviations, Unicode text, and address ordering/formatting.",
            "Exact-matching risks: spelling noise, abbreviations, punctuation, transliteration, reordered addresses, and missing addresses can cause false negatives.",
            "Precision risk: common business names and shared address fragments can create false candidate pairs, so country and multiple blocking signals should be considered.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    for path in [*SOURCE_FILES.values(), GROUND_TRUTH_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required training TSV is missing: {path}")

    ground_truth = collect_ground_truth()
    source_results = {
        label: analyze_source(
            path,
            label,
            (
                set(ground_truth["source1_ids"])
                if label == "Source 1"
                else {
                    entity_id
                    for entity_id in ground_truth["matched_ids"]
                    if entity_id.startswith(f"S{label[-1]}-")
                }
            ),
        )
        for label, path in SOURCE_FILES.items()
    }
    report = write_report(ground_truth, source_results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved analysis report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()