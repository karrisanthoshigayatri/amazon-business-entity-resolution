from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"


def list_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Expected dataset directory does not exist: {directory}"
        )

    files = sorted(path for path in directory.iterdir() if path.is_file())
    print(f"\nFiles in {directory.relative_to(PROJECT_ROOT)}:")
    for path in files:
        print(f"- {path.name}")
    return files


def prefix_for_file(path: Path) -> str | None:
    if "source1" in path.stem:
        return "S1-"
    if "source2" in path.stem:
        return "S2-"
    if "source3" in path.stem:
        return "S3-"
    return None


def report_prefixes(data: pd.DataFrame, path: Path) -> None:
    expected_prefix = prefix_for_file(path)
    if expected_prefix is None or "entity_id" not in data.columns:
        return

    invalid = data.loc[
        ~data["entity_id"].str.startswith(expected_prefix), "entity_id"
    ]
    print(
        f"ID prefix check: expected {expected_prefix}; "
        f"invalid count={len(invalid)}; examples={invalid.head(5).tolist()}"
    )


def inspect_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    print(f"\n{'=' * 72}\nFILE: {path.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Column names: {data.columns.tolist()}")
    print(f"Data types:\n{data.dtypes}")
    print("First 5 rows:")
    print(data.head(5).to_string(index=False))
    print(f"Missing values:\n{data.eq('').sum()}")

    if "entity_id" in data.columns:
        duplicate_ids = data.loc[
            data["entity_id"].duplicated(keep=False), "entity_id"
        ]
        duplicate_counts = duplicate_ids.value_counts()
        print(
            "Duplicate entity_id values: "
            f"{len(duplicate_counts)}; examples={duplicate_counts.head(5).to_dict()}"
        )

    if "country" in data.columns:
        print(f"Unique countries: {data['country'].nunique()}")
        print(f"Country counts:\n{data['country'].value_counts(dropna=False)}")

    report_prefixes(data, path)
    return data


def inspect_ground_truth(path: Path) -> None:
    data = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    matched = data["matched_entity_ids"].str.strip()
    match_counts = Counter()

    for value in matched:
        if not value:
            match_counts["empty"] += 1
        else:
            count = len([item for item in value.split(",") if item])
            match_counts["single" if count == 1 else "multiple"] += 1

    invalid_source1 = data.loc[
        ~data["source1_entity_id"].str.startswith("S1-"), "source1_entity_id"
    ]
    matched_ids = matched[matched != ""].str.split(",").explode()
    invalid_matches = matched_ids.loc[
        ~matched_ids.str.startswith(("S2-", "S3-"))
    ]

    print(f"\n{'=' * 72}\nGROUND TRUTH: {path.relative_to(PROJECT_ROOT)}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Rows: {len(data)}")
    print("First 10 rows:")
    print(data.head(10).to_string(index=False))
    print("matched_entity_ids representation: comma-separated S2-/S3- IDs")
    print(f"Empty matches: {match_counts['empty']}")
    print(f"Single matches: {match_counts['single']}")
    print(f"Multiple matches: {match_counts['multiple']}")
    print(
        "Ground-truth source1_entity_id prefix check: "
        f"invalid count={len(invalid_source1)}; examples={invalid_source1.head(5).tolist()}"
    )
    print(
        "Ground-truth matched ID prefix check: "
        f"invalid count={len(invalid_matches)}; examples={invalid_matches.head(5).tolist()}"
    )


def main() -> None:
    train_files = list_files(TRAIN_DIR)
    test_files = list_files(TEST_DIR)

    all_files = train_files + test_files
    for path in all_files:
        if path.suffix.lower() == ".tsv":
            inspect_file(path)

    ground_truth = TRAIN_DIR / "train_ground_truth.tsv"
    if not ground_truth.is_file():
        raise FileNotFoundError(f"Expected ground-truth file does not exist: {ground_truth}")
    inspect_ground_truth(ground_truth)


if __name__ == "__main__":
    main()