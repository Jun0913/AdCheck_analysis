"""
2차 KoBERT 기반 분석 모듈
- 학습된 KoBERT 모델을 로드하여 문장별 의심도를 분류한다.
- 모델이 없을 경우 1차 규칙기반 결과를 그대로 반환한다.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analysis.models.schemas import SentenceResult, SuspicionLevel

# ────────────────────────────────────────────
# 라벨 매핑: 학습 시 정의한 순서와 반드시 일치해야 함
# 0: 정상, 1: 주의, 2: 의심
# ────────────────────────────────────────────
LABEL_MAP = {
    0: SuspicionLevel.NORMAL,
    1: SuspicionLevel.CAUTION,
    2: SuspicionLevel.SUSPICIOUS,
}

LABEL_REASON_MAP = {
    SuspicionLevel.NORMAL: "KoBERT 모델이 일반적인 광고 표현으로 분류하였습니다.",
    SuspicionLevel.CAUTION: "KoBERT 모델이 과장 가능성이 있는 표현으로 분류하였습니다.",
    SuspicionLevel.SUSPICIOUS: "KoBERT 모델이 허위·과장 의심 표현으로 분류하였습니다.",
}

MODEL_PATH = os.getenv("KOBERT_MODEL_PATH", "models/kobert_ad_classifier")
_tokenizer = None
_model = None


def _load_model():
    """KoBERT 모델 로드 (최초 1회만 실행)"""
    global _tokenizer, _model
    if _model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        return True
    except Exception as e:
        print(f"[KoBERT] 모델 로드 실패: {e}")
        return False


async def analyze_with_kobert(sentence: str, rule_result: SentenceResult) -> SentenceResult:
    """
    2차 KoBERT 분석.
    - 모델이 로드되지 않은 경우 1차 규칙기반 결과를 그대로 반환한다.
    - 모델 예측 결과가 규칙기반 결과보다 높은 의심도이면 모델 결과를 채택한다.
    """
    if not _load_model():
        # 모델 미준비 → 규칙기반 결과 유지
        return rule_result

    try:
        inputs = _tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            pred_label = int(torch.argmax(logits, dim=-1).item())

        kobert_level = LABEL_MAP.get(pred_label, SuspicionLevel.NORMAL)

        # 규칙기반 vs KoBERT 중 더 높은 의심도 채택
        severity = {SuspicionLevel.NORMAL: 0, SuspicionLevel.CAUTION: 1, SuspicionLevel.SUSPICIOUS: 2}
        if severity[kobert_level] >= severity[rule_result.suspicion_level]:
            final_level = kobert_level
            reason = LABEL_REASON_MAP[kobert_level]
        else:
            final_level = rule_result.suspicion_level
            reason = rule_result.reason

        return SentenceResult(
            sentence=sentence,
            suspicion_level=final_level,
            matched_keywords=rule_result.matched_keywords,
            reason=reason,
        )

    except Exception as e:
        print(f"[KoBERT] 추론 오류: {e}")
        return rule_result


def generate_summary(
    overall_level: SuspicionLevel,
    sentence_results: list[SentenceResult],
    score: float,
) -> str:
    """규칙 기반 요약 생성 (외부 API 미사용)"""
    suspicious = [r for r in sentence_results if r.suspicion_level != SuspicionLevel.NORMAL]

    if not suspicious:
        return (
            "해당 광고 문구에서 특별히 의심되는 표현이 발견되지 않았습니다. "
            "본 서비스는 참고용이며 법적 판단을 대신하지 않습니다."
        )

    level_text = {
        SuspicionLevel.CAUTION: "일부 주의가 필요한",
        SuspicionLevel.SUSPICIOUS: "허위·과장 가능성이 높은",
    }
    desc = level_text.get(overall_level, "")
    top = suspicious[0]
    return (
        f"총 {len(sentence_results)}개 문장 중 {len(suspicious)}개에서 {desc} 표현이 감지되었습니다. "
        f"주요 문구: \"{top.sentence}\" — {top.reason} "
        f"광고 내용을 비판적으로 검토하시기 바랍니다."
    )
