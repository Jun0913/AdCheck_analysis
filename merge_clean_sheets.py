"""
Merge training CSV sources into data/merged/merged_clean.csv.

This rebuilds the merged dataset from source files in data/clean instead of
re-appending the existing merged output. Human-reviewed CSVs are normalized
into the training schema and marked so they can be protected later.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Iterable

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
MERGED_DIR = os.path.join(DATA_DIR, "merged")
DEFAULT_OUTPUT = os.path.join(MERGED_DIR, "merged_clean.csv")

EXPECTED_COLUMNS = [
    "정제 ID",
    "원본 ID",
    "정제 문구",
    "키워드",
    "패턴 태그",
    "기준 라벨",
    "의심도 라벨",
    "사유",
]

OPTIONAL_METADATA_COLUMNS = [
    "데이터 출처",
    "사람수집 여부",
]

FULL_OUTPUT_COLUMNS = EXPECTED_COLUMNS + OPTIONAL_METADATA_COLUMNS

CLEAN_FILE_PATTERN = r"^clean_[A-D]_\d{3}\.csv$"
HUMAN_REVIEWED_FILE_PATTERN = r"^human_reviewed.*\.csv$"
DEDUP_MODES = ("none", "id-only", "exact", "normalized")


def is_clean_file(filename: str) -> bool:
    return bool(re.match(CLEAN_FILE_PATTERN, os.path.basename(filename)))


def is_human_reviewed_file(filename: str) -> bool:
    return bool(re.match(HUMAN_REVIEWED_FILE_PATTERN, os.path.basename(filename)))


def find_source_files(directory: str) -> list[str]:
    all_csvs = glob.glob(os.path.join(directory, "*.csv"))
    selected = [
        path for path in all_csvs
        if is_clean_file(path) or is_human_reviewed_file(path)
    ]
    return sorted(selected)


def read_csv_flex(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding="cp949", dtype=str)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s\W_]+", "", text).lower()


def normalize_human_reviewed(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required = ["정제 ID", "원본 ID", "정제 문구", "의심도 라벨"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")

    normalized = pd.DataFrame(
        {
            "정제 ID": df["정제 ID"],
            "원본 ID": df["원본 ID"],
            "정제 문구": df["정제 문구"],
            "키워드": "NA",
            "패턴 태그": "NA",
            "기준 라벨": "NA",
            "의심도 라벨": df["의심도 라벨"],
            "사유": "[human_reviewed_import] confidence=human",
            "데이터 출처": source_name,
            "사람수집 여부": "Y",
        }
    )
    return normalized


def finalize_metadata(df: pd.DataFrame, source_name: str, human_reviewed: bool) -> pd.DataFrame:
    df = df.copy()

    if "데이터 출처" not in df.columns:
        df["데이터 출처"] = source_name
    else:
        df["데이터 출처"] = df["데이터 출처"].fillna(source_name).replace("", source_name)

    default_human = "Y" if human_reviewed else "N"
    if "사람수집 여부" not in df.columns:
        df["사람수집 여부"] = default_human
    else:
        df["사람수집 여부"] = df["사람수집 여부"].fillna(default_human).replace("", default_human)

    return df


def load_source_file(filepath: str) -> tuple[pd.DataFrame | None, list[str]]:
    warnings: list[str] = []
    source_name = os.path.basename(filepath)
    human_reviewed = is_human_reviewed_file(source_name)

    try:
        df = read_csv_flex(filepath)
    except Exception as exc:
        return None, [f"[error] failed to read '{source_name}': {exc}"]

    df.columns = df.columns.str.strip()

    try:
        if human_reviewed:
            df = normalize_human_reviewed(df, source_name)
        else:
            missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing:
                warnings.append(f"[schema] '{source_name}' missing columns: {missing}")
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    df[col] = "NA"
    except ValueError as exc:
        return None, [f"[schema] {exc}"]

    df = finalize_metadata(df, source_name, human_reviewed)
    df["_source_file"] = source_name
    return df, warnings


def check_id_integrity(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if "정제 ID" not in df.columns:
        return issues

    for source, group in df.groupby("_source_file"):
        dupes = group[group["정제 ID"].duplicated(keep=False)]["정제 ID"].dropna().unique()
        if len(dupes) > 0:
            issues.append(f"[duplicate-id] '{source}' has duplicate 정제 ID values: {list(dupes[:10])}")

    global_dupes = df[df["정제 ID"].duplicated(keep=False)]["정제 ID"].dropna().unique()
    if len(global_dupes) > 0:
        preview = list(global_dupes[:10])
        issues.append(f"[duplicate-id] duplicate IDs across files: {preview}")
    return issues


def dedup_by_exact_content(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "정제 문구" not in df.columns:
        return df, 0

    before = len(df)
    df = df.copy()
    mask_empty = df["정제 문구"].fillna("").str.strip() == ""
    valid = df[~mask_empty].drop_duplicates(subset=["정제 문구"], keep="first")
    empty = df[mask_empty]
    return pd.concat([valid, empty], ignore_index=True), before - len(valid) - len(empty) + len(empty)


def dedup_by_normalized_content(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "정제 문구" not in df.columns:
        return df, 0

    before = len(df)
    df = df.copy()
    df["_normalized"] = df["정제 문구"].apply(normalize_text)
    mask_empty = df["_normalized"] == ""
    valid = df[~mask_empty].drop_duplicates(subset=["_normalized"], keep="first")
    empty = df[mask_empty]
    merged = pd.concat([valid, empty], ignore_index=True).drop(columns=["_normalized"])
    return merged, before - len(merged)


def source_priority(source_name: str) -> int:
    if source_name == "human_reviewed_dataset2.csv":
        return 0
    if source_name == "human_reviewed_dataset.csv":
        return 1
    if source_name == "human_reviewed_train_638.csv":
        return 2
    if source_name.startswith("human_reviewed"):
        return 3
    return 10


def row_priority_tuple(row: pd.Series) -> tuple[int, int, int]:
    human_priority = 0 if str(row.get("사람수집 여부", "N")) == "Y" else 1
    source_name = str(row.get("_source_file", ""))
    source_rank = source_priority(source_name)

    filled_fields = 0
    for col in ("키워드", "패턴 태그", "기준 라벨", "사유"):
        value = str(row.get(col, "NA"))
        if value and value != "NA":
            filled_fields += 1

    # Lower tuple wins.
    return (human_priority, source_rank, -filled_fields)


def merge_duplicate_group(group: pd.DataFrame) -> pd.Series:
    sorted_group = group.copy()
    sorted_group["_priority"] = sorted_group.apply(row_priority_tuple, axis=1)
    sorted_group = sorted_group.sort_values("_priority", kind="stable")

    base = sorted_group.iloc[0].copy()
    for _, row in sorted_group.iloc[1:].iterrows():
        for col in sorted_group.columns:
            if col == "_priority":
                continue
            current = str(base.get(col, "NA"))
            candidate = str(row.get(col, "NA"))
            if (not current or current == "NA") and candidate and candidate != "NA":
                base[col] = candidate

    if "_priority" in base.index:
        base = base.drop(labels="_priority")
    return base


def dedup_by_id_with_priority(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "정제 ID" not in df.columns:
        return df, 0

    before = len(df)
    merged_rows = []
    for _, group in df.groupby("정제 ID", sort=False, dropna=False):
        merged_rows.append(merge_duplicate_group(group))
    deduped = pd.DataFrame(merged_rows)
    return deduped, before - len(deduped)


def apply_dedup(df: pd.DataFrame, dedup_mode: str) -> tuple[pd.DataFrame, list[str]]:
    if dedup_mode not in DEDUP_MODES:
        raise ValueError(f"unknown dedup mode: {dedup_mode}")

    warnings: list[str] = []

    if dedup_mode == "none":
        return df, warnings

    if "정제 ID" in df.columns:
        df, removed = dedup_by_id_with_priority(df)
        if removed > 0:
            warnings.append(f"[dedup-id] removed duplicate IDs with human-priority merge: {removed}")

    if dedup_mode == "id-only":
        return df, warnings

    if dedup_mode == "exact":
        df, removed = dedup_by_exact_content(df)
        if removed > 0:
            warnings.append(f"[dedup-text-exact] removed duplicate 문구: {removed}")
        return df, warnings

    df, removed = dedup_by_normalized_content(df)
    if removed > 0:
        warnings.append(f"[dedup-text-normalized] removed normalized duplicate 문구: {removed}")
    return df, warnings


def drop_augmented_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "원본 ID" not in df.columns:
        return df, 0
    mask = df["원본 ID"].fillna("").str.startswith("RAW-AUG-")
    removed = int(mask.sum())
    if removed == 0:
        return df, 0
    return df.loc[~mask].copy(), removed


def exclude_holdout_rows(df: pd.DataFrame, holdout_path: str) -> tuple[pd.DataFrame, int]:
    if not holdout_path or not os.path.exists(holdout_path):
        return df, 0

    holdout = pd.read_csv(holdout_path, encoding="utf-8-sig", dtype=str)
    holdout.columns = holdout.columns.str.strip()
    removed_before = len(df)

    if "정제 ID" in holdout.columns and "정제 ID" in df.columns:
        holdout_ids = set(holdout["정제 ID"].dropna())
        df = df.loc[~df["정제 ID"].isin(holdout_ids)].copy()

    if "정제 문구" in holdout.columns and "정제 문구" in df.columns:
        holdout_texts = set(holdout["정제 문구"].fillna("").str.strip()) - {""}
        df = df.loc[~df["정제 문구"].fillna("").str.strip().isin(holdout_texts)].copy()

    return df, removed_before - len(df)


def order_columns(df: pd.DataFrame, keep_source_col: bool) -> pd.DataFrame:
    ordered = [col for col in FULL_OUTPUT_COLUMNS if col in df.columns]
    extras = [col for col in df.columns if col not in FULL_OUTPUT_COLUMNS]
    if not keep_source_col and "_source_file" in extras:
        extras.remove("_source_file")
    return df[ordered + extras]


def print_summary(merged: pd.DataFrame, source_counts: dict[str, int], dedup_mode: str) -> None:
    print("\n" + "=" * 60)
    print("Merge Summary")
    print("=" * 60)
    print(f"source files: {len(source_counts)}")
    print(f"dedup mode: {dedup_mode}")
    for fname, count in source_counts.items():
        print(f"  {fname}: {count}")

    print(f"\nmerged rows: {len(merged)}")

    for col in ("의심도 라벨", "사람수집 여부", "데이터 출처"):
        if col in merged.columns:
            print(f"\n{col}:")
            print(merged[col].fillna("NA").value_counts(dropna=False).head(20).to_string())


def merge_clean_sheets(
    directory: str = CLEAN_DIR,
    output_path: str | None = None,
    dedup_mode: str = "id-only",
    keep_source_col: bool = False,
    drop_augmented: bool = True,
    exclude_holdout_path: str | None = None,
) -> pd.DataFrame | None:
    print(f"\nsource directory: {os.path.abspath(directory)}")
    files = find_source_files(directory)
    if not files:
        print("No source CSV files found.")
        return None

    print(f"source files found: {len(files)}")
    for filepath in files:
        print(f"  - {os.path.basename(filepath)}")

    all_dfs: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    source_counts: dict[str, int] = {}

    for filepath in files:
        df, warnings = load_source_file(filepath)
        all_warnings.extend(warnings)
        if df is not None:
            all_dfs.append(df)
            source_counts[os.path.basename(filepath)] = len(df)

    if not all_dfs:
        print("No readable source CSV files found.")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    all_warnings.extend(check_id_integrity(merged))

    if drop_augmented:
        merged, removed = drop_augmented_rows(merged)
        if removed > 0:
            all_warnings.append(f"[filter] removed augmented template rows: {removed}")

    if exclude_holdout_path:
        merged, removed = exclude_holdout_rows(merged, exclude_holdout_path)
        if removed > 0:
            all_warnings.append(f"[filter] removed holdout-overlap rows: {removed}")

    merged, dedup_warnings = apply_dedup(merged, dedup_mode)
    all_warnings.extend(dedup_warnings)

    merged = order_columns(merged, keep_source_col=keep_source_col)
    merged = merged.fillna("NA").replace("", "NA")

    if all_warnings:
        print("\nwarnings:")
        for warning in all_warnings:
            print(f"  {warning}")

    print_summary(merged, source_counts, dedup_mode)

    os.makedirs(MERGED_DIR, exist_ok=True)
    final_output = output_path or DEFAULT_OUTPUT
    merged.to_csv(final_output, index=False, encoding="utf-8-sig")
    print(f"\nsaved: {os.path.abspath(final_output)}")
    print("=" * 60 + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge clean and human-reviewed CSV files.")
    parser.add_argument("-d", "--directory", default=CLEAN_DIR, help="source directory")
    parser.add_argument("-o", "--output", default=None, help="output csv path")
    parser.add_argument(
        "--dedup-mode",
        choices=DEDUP_MODES,
        default="id-only",
        help="deduplication mode",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="keep _source_file column in output",
    )
    parser.add_argument(
        "--keep-augmented",
        action="store_true",
        help="keep RAW-AUG template rows instead of removing them",
    )
    parser.add_argument(
        "--exclude-holdout",
        default=os.path.join(DATA_DIR, "eval", "human_reviewed_holdout_200.csv"),
        help="optional holdout csv to exclude from merged data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_clean_sheets(
        directory=args.directory,
        output_path=args.output,
        dedup_mode=args.dedup_mode,
        keep_source_col=args.keep_source,
        drop_augmented=not args.keep_augmented,
        exclude_holdout_path=args.exclude_holdout,
    )
