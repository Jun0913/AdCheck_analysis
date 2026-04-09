"""
정제 시트(clean_*.csv) 병합 도구
딱 걸렸어! 프로젝트 - 허위·과장 광고 의심도 분석 서비스

폴더 구조:
    data/
    ├── raw/        ← 원본 시트 (raw_*.csv) 보관
    ├── clean/      ← 정제 시트 (clean_*.csv) 여기에 넣으세요!
    ├── merged/     ← 병합 결과 CSV 자동 저장
    └── training/   ← 모델 학습용 최종 데이터

사용법:
    python merge_clean_sheets.py                        # 기본 실행 (data/clean/ 자동 탐색)
    python merge_clean_sheets.py -d ./data/clean        # 폴더 직접 지정
    python merge_clean_sheets.py -o output.csv          # 출력 파일명 지정
    python merge_clean_sheets.py --no-dedup             # 중복 제거 비활성화
    python merge_clean_sheets.py --keep-source          # 출처 파일 컬럼 유지
"""

import os
import re
import glob
import argparse
import pandas as pd

# ────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────

# 이 스크립트 기준 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CLEAN_DIR    = os.path.join(DATA_DIR, "clean")    # 정제 시트 입력 폴더
MERGED_DIR   = os.path.join(DATA_DIR, "merged")   # 병합 결과 출력 폴더


# ────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────

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


# ────────────────────────────────────────────
# 유틸 함수
# ────────────────────────────────────────────

def is_clean_file(filename: str) -> bool:
    """파일명이 clean_{작업자ID}_{문서번호3자리}.csv 형식인지 확인"""
    return bool(re.match(CLEAN_FILE_PATTERN, os.path.basename(filename)))


def find_clean_files(directory: str) -> list[str]:
    """지정 폴더에서 정제 시트 CSV 파일 목록 탐색"""
    all_csvs = glob.glob(os.path.join(directory, "*.csv"))
    matched = sorted([f for f in all_csvs if is_clean_file(f)])
    return matched


def load_and_validate(filepath: str) -> tuple[pd.DataFrame | None, list[str]]:
    """
    CSV를 불러오고 컬럼 구조를 검증한다.
    반환: (DataFrame 또는 None, 경고 메시지 목록)
    """
    warnings = []

    try:
        df = pd.read_csv(filepath, encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="cp949", dtype=str)
            warnings.append(f"[인코딩] '{os.path.basename(filepath)}' — cp949로 읽음 (UTF-8 권장)")
        except Exception as e:
            return None, [f"[오류] '{os.path.basename(filepath)}' 읽기 실패: {e}"]
    except Exception as e:
        return None, [f"[오류] '{os.path.basename(filepath)}' 읽기 실패: {e}"]

    # 컬럼 공백 정리
    df.columns = df.columns.str.strip()

    # 필수 컬럼 누락 확인
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        warnings.append(f"[컬럼 누락] '{os.path.basename(filepath)}' — 누락 컬럼: {missing}")

    # 예상 외 컬럼 확인
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        warnings.append(f"[추가 컬럼] '{os.path.basename(filepath)}' — 예상 외 컬럼 존재: {extra}")

    # 소스 파일명 추적 컬럼 추가
    df["_source_file"] = os.path.basename(filepath)

    return df, warnings


def check_id_integrity(df: pd.DataFrame) -> list[str]:
    """정제 ID 무결성 검사"""
    issues = []

    if "정제 ID" not in df.columns:
        return issues

    # 파일 내 중복 정제 ID 확인
    for source, group in df.groupby("_source_file"):
        dupes = group[group["정제 ID"].duplicated(keep=False)]["정제 ID"].dropna().unique()
        if len(dupes) > 0:
            issues.append(f"[중복 ID] '{source}' 내 중복 정제 ID: {list(dupes)}")

    # 전체 파일 통합 후 중복 ID 확인
    global_dupes = df[df["정제 ID"].duplicated(keep=False)]["정제 ID"].dropna().unique()
    if len(global_dupes) > 0:
        issues.append(f"[전체 중복 ID] 파일 간 중복 정제 ID ({len(global_dupes)}건): {list(global_dupes[:5])}{'...' if len(global_dupes) > 5 else ''}")

    return issues


def normalize_text(text: str) -> str:
    """공백·특수문자 제거 후 소문자 변환 — 중복 문구 비교용"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\s\W_]+", "", text)  # 공백, 특수문자, 언더스코어 제거
    return text.lower()


def dedup_by_content(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    정제 문구 기준 중복 제거.
    공백·특수문자 제거 후 표현이 동일한 행은 첫 번째만 유지한다.
    """
    if "정제 문구" not in df.columns:
        return df, 0

    before = len(df)
    df = df.copy()
    df["_normalized"] = df["정제 문구"].apply(normalize_text)

    # 정규화 결과가 빈 문자열인 행(NA 등)은 중복 제거 대상에서 제외
    mask_empty = df["_normalized"] == ""
    df_valid = df[~mask_empty]
    df_empty = df[mask_empty]

    df_valid = df_valid.drop_duplicates(subset=["_normalized"], keep="first")
    df = pd.concat([df_valid, df_empty], ignore_index=True)
    df = df.drop(columns=["_normalized"])

    removed = before - len(df)
    return df, removed


def print_summary(merged: pd.DataFrame, source_counts: dict):
    """병합 결과 요약 출력"""
    print("\n" + "=" * 50)
    print("📊 병합 결과 요약")
    print("=" * 50)

    print(f"\n📁 파일별 행 수:")
    for fname, cnt in source_counts.items():
        print(f"   {fname}: {cnt}행")

    print(f"\n📝 총 병합 행 수: {len(merged)}행 (중복 제거 후)")

    if "기준 라벨" in merged.columns:
        label_counts = merged["기준 라벨"].value_counts(dropna=False)
        print(f"\n🏷️  기준 라벨 분포:")
        for label, cnt in label_counts.items():
            print(f"   {label}: {cnt}건")

    if "의심도 라벨" in merged.columns:
        suspicion_counts = merged["의심도 라벨"].value_counts(dropna=False)
        print(f"\n⚠️  의심도 라벨 분포:")
        for label, cnt in suspicion_counts.items():
            print(f"   {label}: {cnt}건")

    if "패턴 태그" in merged.columns:
        all_tags = (
            merged["패턴 태그"]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
        )
        print(f"\n🔖 패턴 태그 분포 (상위 10개):")
        for tag, cnt in all_tags.head(10).items():
            print(f"   {tag}: {cnt}건")

    print()


# ────────────────────────────────────────────
# 메인 로직
# ────────────────────────────────────────────

def merge_clean_sheets(
    directory: str = CLEAN_DIR,
    output_path: str = None,
    dedup: bool = True,
    keep_source_col: bool = False,
) -> pd.DataFrame | None:

    print(f"\n🔍 탐색 경로: {os.path.abspath(directory)}")

    files = find_clean_files(directory)

    if not files:
        print("❌ 정제 시트 파일(clean_*.csv)을 찾을 수 없습니다.")
        return None

    print(f"✅ 발견된 정제 시트: {len(files)}개")
    for f in files:
        print(f"   - {os.path.basename(f)}")

    all_dfs = []
    all_warnings = []
    source_counts = {}

    for filepath in files:
        df, warnings = load_and_validate(filepath)
        all_warnings.extend(warnings)
        if df is not None:
            source_counts[os.path.basename(filepath)] = len(df)
            all_dfs.append(df)

    if not all_dfs:
        print("❌ 읽을 수 있는 파일이 없습니다.")
        return None

    # 병합
    merged = pd.concat(all_dfs, ignore_index=True)

    # 컬럼 순서 정렬
    ordered_cols = [c for c in EXPECTED_COLUMNS if c in merged.columns]
    extra_cols = [c for c in merged.columns if c not in EXPECTED_COLUMNS]
    if not keep_source_col and "_source_file" in extra_cols:
        extra_cols.remove("_source_file")
    merged = merged[ordered_cols + extra_cols]

    # ID 무결성 검사
    id_issues = check_id_integrity(
        pd.concat(all_dfs, ignore_index=True)
    )
    all_warnings.extend(id_issues)

    # 1단계: 정제 ID 기준 중복 제거
    before_dedup = len(merged)
    if dedup and "정제 ID" in merged.columns:
        merged = merged.drop_duplicates(subset=["정제 ID"], keep="first")
        removed_id = before_dedup - len(merged)
        if removed_id > 0:
            all_warnings.append(f"[중복 제거 - ID] 정제 ID 기준 중복 {removed_id}행 제거됨")

    # 2단계: 정제 문구 내용 기준 중복 제거 (공백·특수문자 제거 후 표현 동일한 경우)
    if dedup:
        merged, removed_content = dedup_by_content(merged)
        if removed_content > 0:
            all_warnings.append(f"[중복 제거 - 문구] 동일 표현 중복 {removed_content}행 제거됨")

    # NA 정규화
    merged = merged.fillna("NA").replace("", "NA")

    # 경고 출력
    if all_warnings:
        print(f"\n⚠️  검증 경고 ({len(all_warnings)}건):")
        for w in all_warnings:
            print(f"   {w}")

    # 요약 출력
    print_summary(merged, source_counts)

    # 출력 파일 저장
    os.makedirs(MERGED_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(MERGED_DIR, "merged_clean.csv")

    # 기존 merged_clean.csv가 있으면 추가 병합
    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str)
            existing.columns = existing.columns.str.strip()
            existing_count = len(existing)
            merged = pd.concat([existing, merged], ignore_index=True)
            print(f"📂 기존 파일 감지: {existing_count}행 + 새 데이터 추가 후 중복 제거 진행")

            # 컬럼 순서 재정렬
            ordered_cols = [c for c in EXPECTED_COLUMNS if c in merged.columns]
            extra_cols = [c for c in merged.columns if c not in EXPECTED_COLUMNS]
            if not keep_source_col and "_source_file" in extra_cols:
                extra_cols.remove("_source_file")
            merged = merged[ordered_cols + extra_cols]

            # 중복 재제거
            if dedup and "정제 ID" in merged.columns:
                before = len(merged)
                merged = merged.drop_duplicates(subset=["정제 ID"], keep="first")
                removed = before - len(merged)
                if removed > 0:
                    print(f"   [중복 제거 - ID] {removed}행 제거")
            if dedup:
                merged, removed_content = dedup_by_content(merged)
                if removed_content > 0:
                    print(f"   [중복 제거 - 문구] {removed_content}행 제거")

            # NA 재정규화
            merged = merged.fillna("NA").replace("", "NA")
        except Exception as e:
            print(f"⚠️  기존 파일 읽기 실패, 새로 생성합니다: {e}")

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"💾 저장 완료: {os.path.abspath(output_path)}")
    print("=" * 50 + "\n")

    return merged


# ────────────────────────────────────────────
# CLI 진입점
# ────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="정제 시트(clean_*.csv) 병합 도구 — 딱 걸렸어! 프로젝트"
    )
    parser.add_argument(
        "-d", "--directory",
        default=CLEAN_DIR,
        help=f"정제 시트가 있는 폴더 경로 (기본값: data/clean/)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="출력 파일명 (기본값: merged_clean_YYYYMMDD_HHMMSS.csv)"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="중복 제거 비활성화 (동일 정제 ID 행도 모두 유지)"
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="결과에 _source_file 컬럼 유지 (어느 파일에서 왔는지 추적)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_clean_sheets(
        directory=args.directory,
        output_path=args.output,
        dedup=not args.no_dedup,
        keep_source_col=args.keep_source,
    )
