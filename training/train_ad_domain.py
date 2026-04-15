import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 기본 경로
DEFAULT_DATA = Path("data/merged/merged_clean.csv")
AD_MODEL_OUT = Path("models/ad_filter.joblib")
COSMETIC_MODEL_OUT = Path("models/cosmetic_filter.joblib")


COSMETIC_HINTS = {
    "크림",
    "로션",
    "앰플",
    "에센스",
    "스킨",
    "토너",
    "세럼",
    "클렌저",
    "클렌징",
    "선크림",
    "썬크림",
    "선스틱",
    "팩",
    "마스크",
    "시트마스크",
    "립",
    "틴트",
    "쿠션",
    "파운데이션",
    "베이스",
    "아이섀도우",
    "블러셔",
    "립밤",
    "샴푸",
    "헤어",
    "트리트먼트",
    "에어쿠션",
    "비비",
    "씨씨",
    "모공",
    "여드름",
    "피부",
    "미백",
    "주름",
    "각질",
}


def is_noise(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if len(s) < 3:
        return True
    if len(s) < 6 and len(s.split()) <= 1:
        return True
    alpha_num = sum(ch.isalnum() for ch in s)
    if alpha_num / max(len(s), 1) < 0.3:
        return True
    return False


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    광고/화장품 여부를 자동 파생해 새로운 컬럼을 추가한다.
    - is_noise: 극단적 잡음
    - is_ad: '의심도 라벨'이 '주의' 또는 '의심'이면 1, 그 외 0
    - is_cosmetic: 정제 문구나 키워드에 COSMETIC_HINTS가 하나라도 포함되면 1
    """
    df = df.copy()
    text_col = "정제 문구"
    label_col = "의심도 라벨"
    keyword_col = "키워드"

    df["is_noise"] = df[text_col].apply(is_noise)
    df["is_ad"] = df[label_col].isin(["주의", "의심"]).astype(int)

    def cosmetic_flag(row):
        text = f"{row.get(text_col,'')} {row.get(keyword_col,'')}"
        return int(any(h in text for h in COSMETIC_HINTS))

    df["is_cosmetic"] = df.apply(cosmetic_flag, axis=1)
    return df


def train_lr(series_text, series_label):
    """
    TF-IDF (1~3gram) + LogisticRegression(class_weight='balanced') 파이프라인을 학습 후 반환.
    """
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )
    pipe = make_pipeline(vec, clf)
    pipe.fit(series_text, series_label)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="광고/화장품 전단 필터 학습 스크립트")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="입력 CSV 경로")
    parser.add_argument("--ad-out", type=Path, default=AD_MODEL_OUT, help="광고 모델 저장 경로")
    parser.add_argument(
        "--cosmetic-out", type=Path, default=COSMETIC_MODEL_OUT, help="화장품 모델 저장 경로"
    )
    parser.add_argument("--report", action="store_true", help="검증 리포트 출력")
    parser.add_argument(
        "--export-labeled",
        type=Path,
        default=None,
        help="라벨이 파생된 CSV를 저장할 경로(예: data/clean/ad_domain_labeled.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = derive_labels(df)

    # 파생 라벨 CSV 저장 옵션
    if args.export_labeled:
        os.makedirs(args.export_labeled.parent, exist_ok=True)
        df.to_csv(args.export_labeled, index=False)
        print(f"[info] labeled data saved -> {args.export_labeled}")

    # 잡음은 학습에서 제외
    train_df = df[~df["is_noise"]].copy()

    # 광고 여부 모델
    X_ad = train_df["정제 문구"]
    y_ad = train_df["is_ad"]
    ad_model = train_lr(X_ad, y_ad)

    # 화장품 여부 모델
    X_cos = train_df["정제 문구"]
    y_cos = train_df["is_cosmetic"]
    cos_model = train_lr(X_cos, y_cos)

    # 저장: (vectorizer, classifier) 튜플 형태와 호환되도록
    os.makedirs(args.ad_out.parent, exist_ok=True)
    os.makedirs(args.cosmetic_out.parent, exist_ok=True)
    joblib.dump((ad_model.named_steps["tfidfvectorizer"], ad_model.named_steps["logisticregression"]), args.ad_out)
    joblib.dump(
        (cos_model.named_steps["tfidfvectorizer"], cos_model.named_steps["logisticregression"]), args.cosmetic_out
    )

    if args.report:
        def eval_model(pipe, X, y, name):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            print(f"\n=== {name} ===")
            print(classification_report(y_te, y_pred, digits=3))

        eval_model(ad_model, X_ad, y_ad, "AD")
        eval_model(cos_model, X_cos, y_cos, "COSMETIC")

    print(f"[done] saved ad model -> {args.ad_out}")
    print(f"[done] saved cosmetic model -> {args.cosmetic_out}")


if __name__ == "__main__":
    main()
