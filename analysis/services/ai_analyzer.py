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

# 패턴 태그 → 한국어 설명 (rule_engine과 동일하게 유지)
_PATTERN_DESC: dict[str, str] = {
    "의약품오인":  "의약품으로 오인될 수 있는 표현",
    "효능과장":   "효능을 과장하거나 절대적으로 단정하는 표현",
    "기능성오인":  "기능성 화장품 심사 없이 기능성을 주장하는 표현",
    "안전성단정":  "안전성을 근거 없이 단정하는 표현",
    "추천보증":   "전문가 추천·인증을 주장하는 표현",
    "비교우위":   "근거 없이 타사 대비 우위를 주장하는 표현",
}

def _build_kobert_reason(level: SuspicionLevel, rule_result: SentenceResult, weighted_score: float) -> str:
    """KoBERT가 규칙기반보다 높은 의심도를 채택할 때 이유 생성"""
    keywords = rule_result.matched_keywords

    if level == SuspicionLevel.NORMAL:
        return "일반적인 광고 표현으로 분류되었습니다."

    if keywords:
        kw_display = ", ".join(f"'{kw}'" for kw in keywords[:2])
        if level == SuspicionLevel.SUSPICIOUS:
            return (
                f"AI 분석 결과 {kw_display} 등의 표현이 문장 전체 맥락에서 "
                f"소비자를 오인하게 할 가능성이 높은 허위·과장 표현으로 판단됩니다."
            )
        else:
            return (
                f"AI 분석 결과 {kw_display} 등의 표현이 문장 전체 맥락에서 "
                f"과장 가능성이 있는 표현으로 판단됩니다."
            )

    # 키워드 없이 KoBERT만으로 잡힌 경우 — 문장에서 핵심 어절 추출
    words = [w for w in rule_result.sentence.split() if len(w) > 1]
    sample = ", ".join(f"'{w}'" for w in words[:3]) if words else "해당 문구"
    if level == SuspicionLevel.SUSPICIOUS:
        return (
            f"AI 모델이 문장 전체 맥락을 분석한 결과 {sample} 등의 표현 방식이 "
            f"소비자를 오인하게 할 수 있는 허위·과장 표현으로 판단됩니다."
        )
    else:
        return (
            f"AI 모델이 문장 전체 맥락을 분석한 결과 {sample} 등의 표현 방식이 "
            f"과장 가능성이 있어 주의가 필요합니다."
        )

MODEL_PATH = os.getenv("KOBERT_MODEL_PATH", "models/kobert_ad_classifier")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        _model.eval()
        return True
    except Exception as e:
        print(f"[KoBERT] 모델 로드 실패: {e}")
        return False


async def analyze_with_kobert(sentence: str, rule_result: SentenceResult) -> SentenceResult:
    """
    2차 KoBERT 분석.
    - 규칙기반 결과와 관계없이 모든 문장을 KoBERT로 최종 판단
    - 규칙기반은 키워드/패턴 탐지 역할, KoBERT는 문맥 기반 최종 의심도 결정
    - 단, KoBERT 모델이 없으면 규칙기반 결과를 그대로 반환
    """
    if not _load_model():
        return rule_result

    try:
        inputs = _tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]  # [정상, 주의, 의심] 확률

        prob_normal     = probs[0].item()
        prob_caution    = probs[1].item()
        prob_suspicious = probs[2].item()

        # ── 확률 가중 점수 계산 ───────────────────────────────
        # 정상=0, 주의=0.5, 의심=1.0 으로 가중합산
        # 각 라벨 확률을 반영해 부드럽게 의심도 산출
        weighted_score = (prob_caution * 0.5) + (prob_suspicious * 1.0)

        # ── 가중 점수 → 의심도 레벨 변환 ────────────────────
        # 규칙기반 결과에 따라 임계값 차등 적용
        # - 규칙기반이 의심: KoBERT가 확인/하향 조정
        # - 규칙기반이 주의: KoBERT가 올리거나 내릴 수 있음
        # - 규칙기반이 정상: KoBERT 임계값 높여서 과탐지 방지
        if rule_result.suspicion_level == SuspicionLevel.SUSPICIOUS:
            # 규칙기반이 의심 → KoBERT가 정상이어도 최소 주의 유지
            if weighted_score >= 0.40:
                kobert_level = SuspicionLevel.SUSPICIOUS
            else:
                kobert_level = SuspicionLevel.CAUTION
        elif rule_result.suspicion_level == SuspicionLevel.CAUTION:
            # 규칙기반이 주의 → KoBERT가 올릴 수도 내릴 수도 있음
            if weighted_score >= 0.55:
                kobert_level = SuspicionLevel.SUSPICIOUS
            elif weighted_score >= 0.20:
                kobert_level = SuspicionLevel.CAUTION
            else:
                kobert_level = SuspicionLevel.NORMAL
        else:
            # 규칙기반이 정상 → KoBERT 임계값 높게 설정 (과탐지 방지)
            if weighted_score >= 0.70:
                kobert_level = SuspicionLevel.SUSPICIOUS
            elif weighted_score >= 0.45:
                kobert_level = SuspicionLevel.CAUTION
            else:
                kobert_level = SuspicionLevel.NORMAL

        # ── 최종 결과 반환 ────────────────────────────────────
        # KoBERT 결과가 규칙기반과 다르면 KoBERT 결과 채택 + 이유 생성
        # KoBERT 결과가 같으면 규칙기반 이유 그대로 유지
        if kobert_level != rule_result.suspicion_level:
            reason = _build_kobert_reason(kobert_level, rule_result, weighted_score)
            print(f"[KoBERT] 룰엔진:{rule_result.suspicion_level.name} -> KoBERT:{kobert_level.name} ({weighted_score:.2f})")
        else:
            reason = rule_result.reason

        return SentenceResult(
            sentence=sentence,
            suspicion_level=kobert_level,
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
