"""
Merge cleaned CSV sheets for training.

Usage:
    python merge_clean_sheets.py
    python merge_clean_sheets.py -d ./data/clean
    python merge_clean_sheets.py -o output.csv
    python merge_clean_sheets.py --dedup-mode id-only
    python merge_clean_sheets.py --dedup-mode exact
    python merge_clean_sheets.py --dedup-mode normalized
    python merge_clean_sheets.py --dedup-mode none
    python merge_clean_sheets.py --keep-source
"""

import argparse
import glob
import os
import re

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
MERGED_DIR = os.path.join(DATA_DIR, "merged")

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

CLEAN_FILE_PATTERN = r"^clean_[A-D]_\d{3}\.csv$"
DEDUP_MODES = ("none", "id-only", "exact", "normalized")


def is_clean_file(filename: str) -> bool:
    return bool(re.match(CLEAN_FILE_PATTERN, os.path.basename(filename)))


def find_clean_files(directory: str) -> list[str]:
    all_csvs = glob.glob(os.path.join(directory, "*.csv"))
    return sorted([f for f in all_csvs if is_clean_file(f)])


def load_and_validate(filepath: str) -> tuple[pd.DataFrame | None, list[str]]:
    warnings = []

    try:
        df = pd.read_csv(filepath, encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="cp949", dtype=str)
            warnings.append(
                f"[인코딩] '{os.path.basename(filepath)}' 을 cp949로 읽음 (UTF-8 권장)"
            )
        except Exception as exc:
            return None, [f"[오류] '{os.path.basename(filepath)}' 읽기 실패: {exc}"]
    except Exception as exc:
        return None, [f"[오류] '{os.path.basename(filepath)}' 읽기 실패: {exc}"]

    df.columns = df.columns.str.strip()

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        warnings.append(f"[컬럼 누락] '{os.path.basename(filepath)}' 누락 컬럼: {missing}")

    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra:
        warnings.append(f"[추가 컬럼] '{os.path.basename(filepath)}' 예상 외 컬럼: {extra}")

    df["_source_file"] = os.path.basename(filepath)
    return df, warnings


def check_id_integrity(df: pd.DataFrame) -> list[str]:
    issues = []

    if "정제 ID" not in df.columns:
        return issues

    for source, group in df.groupby("_source_file"):
        dupes = group[group["정제 ID"].duplicated(keep=False)]["정제 ID"].dropna().unique()
        if len(dupes) > 0:
            issues.append(f"[중복 ID] '{source}' 내 중복 정제 ID: {list(dupes)}")

    global_dupes = df[df["정제 ID"].duplicated(keep=False)]["정제 ID"].dropna().unique()
    if len(global_dupes) > 0:
        preview = list(global_dupes[:5])
        suffix = "..." if len(global_dupes) > 5 else ""
        issues.append(
            f"[전체 중복 ID] 파일 간 중복 정제 ID ({len(global_dupes)}개): {preview}{suffix}"
        )

    return issues


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s\W_]+", "", text).lower()


def dedup_by_exact_content(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "정제 문구" not in df.columns:
        return df, 0

    before = len(df)
    df = df.copy()
    mask_empty = df["정제 문구"].fillna("").str.strip() == ""
    df_valid = df[~mask_empty]
    df_empty = df[mask_empty]

    df_valid = df_valid.drop_duplicates(subset=["정제 문구"], keep="first")
    df = pd.concat([df_valid, df_empty], ignore_index=True)
    return df, before - len(df)


def dedup_by_normalized_content(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "정제 문구" not in df.columns:
        return df, 0

    before = len(df)
    df = df.copy()
    df["_normalized"] = df["정제 문구"].apply(normalize_text)

    mask_empty = df["_normalized"] == ""
    df_valid = df[~mask_empty]
    df_empty = df[mask_empty]

    df_valid = df_valid.drop_duplicates(subset=["_normalized"], keep="first")
    df = pd.concat([df_valid, df_empty], ignore_index=True)
    df = df.drop(columns=["_normalized"])
    return df, before - len(df)


def apply_dedup(df: pd.DataFrame, dedup_mode: str) -> tuple[pd.DataFrame, list[str]]:
    if dedup_mode not in DEDUP_MODES:
        raise ValueError(f"Unknown dedup_mode: {dedup_mode}")

    warnings: list[str] = []

    if dedup_mode == "none":
        return df, warnings

    if "정제 ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["정제 ID"], keep="first")
        removed = before - len(df)
        if removed > 0:
            warnings.append(f"[중복 제거 - ID] 정제 ID 기준 중복 {removed}개 제거")

    if dedup_mode == "id-only":
        return df, warnings

    if dedup_mode == "exact":
        df, removed_content = dedup_by_exact_content(df)
        if removed_content > 0:
            warnings.append(f"[중복 제거 - 문구:exact] 완전 동일 문구 {removed_content}개 제거")
        return df, warnings

    df, removed_content = dedup_by_normalized_content(df)
    if removed_content > 0:
        warnings.append(
            f"[중복 제거 - 문구:normalized] 공백/기호 무시 동일 문구 {removed_content}개 제거"
        )
    return df, warnings


def print_summary(merged: pd.DataFrame, source_counts: dict[str, int], dedup_mode: str):
    print("\n" + "=" * 50)
    print("병합 결과 요약")
    print("=" * 50)

    print(f"\n탐색 파일 수: {len(source_counts)}")
    print(f"중복 제거 모드: {dedup_mode}")
    for fname, count in source_counts.items():
        print(f"  {fname}: {count}행")

    print(f"\n총 병합 행 수: {len(merged)}행")

    if "기준 라벨" in merged.columns:
        label_counts = merged["기준 라벨"].value_counts(dropna=False)
        print("\n기준 라벨 분포:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}개")

    if "의심도 라벨" in merged.columns:
        suspicion_counts = merged["의심도 라벨"].value_counts(dropna=False)
        print("\n의심도 라벨 분포:")
        for label, count in suspicion_counts.items():
            print(f"  {label}: {count}개")

    if "패턴 태그" in merged.columns:
        all_tags = (
            merged["패턴 태그"].dropna().str.split(",").explode().str.strip().value_counts()
        )
        print("\n패턴 태그 분포 (상위 10개):")
        for tag, count in all_tags.head(10).items():
            print(f"  {tag}: {count}개")

    print()


def order_columns(df: pd.DataFrame, keep_source_col: bool) -> pd.DataFrame:
    ordered_cols = [col for col in EXPECTED_COLUMNS if col in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if not keep_source_col and "_source_file" in extra_cols:
        extra_cols.remove("_source_file")
    return df[ordered_cols + extra_cols]


def merge_clean_sheets(
    directory: str = CLEAN_DIR,
    output_path: str | None = None,
    dedup_mode: str = "id-only",
    keep_source_col: bool = False,
) -> pd.DataFrame | None:
    print(f"\n탐색 경로: {os.path.abspath(directory)}")

    files = find_clean_files(directory)
    if not files:
        print("정제 시트 파일(clean_*.csv)을 찾을 수 없습니다.")
        return None

    print(f"발견된 정제 시트: {len(files)}개")
    for filepath in files:
        print(f"  - {os.path.basename(filepath)}")

    all_dfs = []
    all_warnings: list[str] = []
    source_counts: dict[str, int] = {}

    for filepath in files:
        df, warnings = load_and_validate(filepath)
        all_warnings.extend(warnings)
        if df is not None:
            source_counts[os.path.basename(filepath)] = len(df)
            all_dfs.append(df)

    if not all_dfs:
        print("읽을 수 있는 파일이 없습니다.")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = order_columns(merged, keep_source_col=keep_source_col)

    id_issues = check_id_integrity(pd.concat(all_dfs, ignore_index=True))
    all_warnings.extend(id_issues)

    merged, dedup_warnings = apply_dedup(merged, dedup_mode)
    all_warnings.extend(dedup_warnings)

    merged = merged.fillna("NA").replace("", "NA")

    if all_warnings:
        print(f"\n검증 경고 ({len(all_warnings)}건):")
        for warning in all_warnings:
            print(f"  {warning}")

    print_summary(merged, source_counts, dedup_mode)

    os.makedirs(MERGED_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(MERGED_DIR, "merged_clean.csv")

    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str)
            existing.columns = existing.columns.str.strip()
            existing_count = len(existing)
            merged = pd.concat([existing, merged], ignore_index=True)
            print(f"기존 병합본 {existing_count}행과 신규 데이터를 다시 병합합니다.")

            merged = order_columns(merged, keep_source_col=keep_source_col)
            merged, existing_dedup_warnings = apply_dedup(merged, dedup_mode)
            for warning in existing_dedup_warnings:
                print(f"  {warning}")

            merged = merged.fillna("NA").replace("", "NA")
        except Exception as exc:
            print(f"기존 파일 읽기 실패로 새 파일로 저장합니다: {exc}")

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {os.path.abspath(output_path)}")
    print("=" * 50 + "\n")
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="정제 시트(clean_*.csv) 병합 도구")
    parser.add_argument(
        "-d",
        "--directory",
        default=CLEAN_DIR,
        help="정제 시트가 있는 폴더 경로 (기본값: data/clean/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="출력 파일명 (기본값: data/merged/merged_clean.csv)",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=DEDUP_MODES,
        default="id-only",
        help="중복 제거 모드: none, id-only, exact, normalized (기본값: id-only)",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="결과에 _source_file 컬럼 유지",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_clean_sheets(
        directory=args.directory,
        output_path=args.output,
        dedup_mode=args.dedup_mode,
        keep_source_col=args.keep_source,
    )
