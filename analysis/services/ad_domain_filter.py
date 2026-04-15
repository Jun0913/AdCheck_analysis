import os
from typing import Optional, Tuple

import numpy as np
import joblib

# 경로와 임계값을 환경변수로 조정 가능
AD_MODEL_PATH = os.getenv("AD_FILTER_MODEL_PATH", "models/ad_filter.joblib")
COSMETIC_MODEL_PATH = os.getenv("COSMETIC_FILTER_MODEL_PATH", "models/cosmetic_filter.joblib")
AD_THRESHOLD = float(os.getenv("AD_FILTER_THRESHOLD", "0.55"))
COSMETIC_THRESHOLD = float(os.getenv("COSMETIC_FILTER_THRESHOLD", "0.55"))


def _load_model(path: str) -> Optional[Tuple[object, object]]:
    """joblib으로 저장된 (vectorizer, classifier) 튜플을 로드한다."""
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, tuple) and len(bundle) == 2:
            return bundle
        # dict 형식 지원
        if isinstance(bundle, dict) and "vectorizer" in bundle and "model" in bundle:
            return bundle["vectorizer"], bundle["model"]
    except Exception as e:
        print(f"[ad_domain_filter] 모델 로드 실패({path}): {e}")
    return None


_ad_bundle = _load_model(AD_MODEL_PATH)
_cosmetic_bundle = _load_model(COSMETIC_MODEL_PATH)


def _predict(text: str, bundle: Optional[Tuple[object, object]], positive_label) -> Optional[Tuple[bool, float]]:
    """주어진 번들(vectorizer, classifier)로 확률 예측."""
    if bundle is None:
        return None
    vectorizer, clf = bundle
    try:
        X = vectorizer.transform([text])
        proba = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        if positive_label in classes:
            idx = classes.index(positive_label)
        elif len(classes) == 2:
            idx = 1  # 양성 클래스가 1개뿐이면 두 번째를 양성으로 가정
        else:
            idx = int(np.argmax(proba))
        score = float(proba[idx])
        return score >= (AD_THRESHOLD if positive_label == "ad" else COSMETIC_THRESHOLD), score
    except Exception as e:
        print(f"[ad_domain_filter] 예측 실패: {e}")
        return None


def predict_ad(text: str) -> Optional[Tuple[bool, float]]:
    """광고 여부 예측. 모델 없으면 None."""
    return _predict(text, _ad_bundle, positive_label="ad")


def predict_cosmetic(text: str) -> Optional[Tuple[bool, float]]:
    """화장품 도메인 여부 예측. 모델 없으면 None."""
    return _predict(text, _cosmetic_bundle, positive_label="cosmetic")


def is_noise(text: str) -> bool:
    """
    극단적으로 짧거나 기호/자모 비율이 높은 잡음 문장을 컷.
    - 길이 < 3
    - 길이 < 6 이면서 공백 포함 단어 수 <= 1
    - 영숫자/한글 비율이 전체의 0.3 미만
    """
    s = text.strip()
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
