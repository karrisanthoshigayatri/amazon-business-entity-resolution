import pandas as pd
from pathlib import Path


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_train_data(train_dir: Path):
    s1 = read_tsv(train_dir / "train_source1.tsv")
    s2 = read_tsv(train_dir / "train_source2.tsv")
    s3 = read_tsv(train_dir / "train_source3.tsv")
    gt = read_tsv(train_dir / "train_ground_truth.tsv")
    return s1, s2, s3, gt


def load_test_data(test_dir: Path):
    s1 = read_tsv(test_dir / "test_source1.tsv")
    s2 = read_tsv(test_dir / "test_source2.tsv")
    s3 = read_tsv(test_dir / "test_source3.tsv")
    return s1, s2, s3
