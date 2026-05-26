"""
2차 KoBERT 기반 분석 모듈
- 학습된 KoBERT 모델을 로드하여 문장별 의심도를 분류한다.
- 모델이 없을 경우 1차 규칙기반 결과를 그대로 반환한다.
"""

import os
import threading
from analysis.services.model_runtime import prepare_model_runtime

prepare_model_runtime()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analysis.models.schemas import SentenceResult, SuspicionLevel
from analysis.services.rule_engine import (
    CONTEXT_ALLOW_NORMAL_REASON,
    DEFAULT_NORMAL_REASON,
    EXTRA_ALLOW_NORMAL_REASON,
    NON_DOMAIN_IMAGE_NORMAL_REASON,
    NON_DOMAIN_NORMAL_REASON,
    STRICT_ALLOW_NORMAL_REASON,
    UNCERTAIN_DOMAIN_REASON,
)
from analysis.services.rule_patterns import (
    CAUTION_ONLY_PATTERNS,
    MIN_PATTERN_LEVELS,
    STRONG_SUSPICIOUS_PATTERNS,
)
from analysis.services.rule_engine import FORBIDDEN_KEYWORDS

# ────────────────────────────────────────────
# 라벨 매핑: 학습 시 정의한 순서와 반드시 일치해야 함
# 0: 정상, 1: 주의, 2: 의심
# ────────────────────────────────────────────
LABEL_MAP = {
    0: SuspicionLevel.NORMAL,
    1: SuspicionLevel.CAUTION,
    2: SuspicionLevel.SUSPICIOUS,
}

_LEVEL_ORDER = {
    SuspicionLevel.NORMAL: 0,
    SuspicionLevel.CAUTION: 1,
    SuspicionLevel.SUSPICIOUS: 2,
}


def _resolve_kobert_level(rule_result: SentenceResult, weighted_score: float) -> SuspicionLevel:
    """KoBERT 점수를 우선하되, 명백한 의심 패턴만 정책상 최소 레벨을 유지한다."""
    if weighted_score >= 0.75:
        return SuspicionLevel.SUSPICIOUS
    if (
        rule_result.suspicion_level == SuspicionLevel.CAUTION
        and len(rule_result.matched_patterns) >= 2
        and weighted_score >= 0.40
    ):
        return SuspicionLevel.SUSPICIOUS
    if weighted_score >= 0.50:
        return SuspicionLevel.CAUTION
    if (
        rule_result.suspicion_level == SuspicionLevel.CAUTION
        and rule_result.score >= 0.35
        and weighted_score >= 0.05
    ):
        return SuspicionLevel.CAUTION
    if (
        rule_result.suspicion_level == SuspicionLevel.CAUTION
        and weighted_score >= 0.20
    ):
        return SuspicionLevel.CAUTION
    if (
        rule_result.suspicion_level == SuspicionLevel.SUSPICIOUS
        and set(rule_result.matched_patterns).intersection(STRONG_SUSPICIOUS_PATTERNS)
        and weighted_score >= 0.18
    ):
        return SuspicionLevel.CAUTION
    return SuspicionLevel.NORMAL

def _build_kobert_reason(level: SuspicionLevel, rule_result: SentenceResult, weighted_score: float) -> str:
    """KoBERT가 규칙기반보다 높은 의심도를 채택할 때 이유 생성"""
    keywords = rule_result.matched_keywords

    if level == SuspicionLevel.NORMAL:
        if rule_result.reason == UNCERTAIN_DOMAIN_REASON:
            return "짧은 광고 문구로 보이지만, 현재 기준에서는 문제 표현이 확인되지 않았습니다."
        return "현재 기준에서는 문제 표현이 확인되지 않았습니다."

    if keywords:
        kw_display = ", ".join(f"'{kw}'" for kw in keywords[:2])
        if level == SuspicionLevel.SUSPICIOUS:
            return (
                f"{kw_display} 같은 표현이 문장 전체 맥락에서 과장되거나 오해를 부를 가능성이 높습니다."
            )
        else:
            return (
                f"{kw_display} 같은 표현이 문장 전체 맥락에서 다소 과장되게 받아들여질 수 있습니다."
            )

    # 키워드 없이 KoBERT만으로 잡힌 경우 - 문장에서 핵심 어절 추출
    words = [w for w in rule_result.sentence.split() if len(w) > 1]
    sample = ", ".join(f"'{w}'" for w in words[:3]) if words else "해당 문구"
    if level == SuspicionLevel.SUSPICIOUS:
        return (
            f"{sample} 같은 표현 방식이 문장 전체 맥락에서 과장되거나 오해를 부를 가능성이 높습니다."
        )
    else:
        return (
            f"{sample} 같은 표현 방식이 다소 과장되게 받아들여질 수 있습니다."
        )


def _enforce_minimum_pattern_level(result: SentenceResult) -> SentenceResult:
    minimum_level = SuspicionLevel.NORMAL
    for pattern in result.matched_patterns:
        pattern_level = MIN_PATTERN_LEVELS.get(pattern, SuspicionLevel.NORMAL)
        if _LEVEL_ORDER[pattern_level] > _LEVEL_ORDER[minimum_level]:
            minimum_level = pattern_level

    minimum_score = {
        SuspicionLevel.CAUTION: 0.35,
        SuspicionLevel.SUSPICIOUS: 0.70,
    }.get(minimum_level, result.score)
    if _LEVEL_ORDER[result.suspicion_level] >= _LEVEL_ORDER[minimum_level]:
        if result.score >= minimum_score:
            return result
        return SentenceResult(
            sentence=result.sentence,
            suspicion_level=result.suspicion_level,
            matched_keywords=result.matched_keywords,
            matched_patterns=result.matched_patterns,
            reason=result.reason,
            score=minimum_score,
        )

    if result.reason in NORMAL_REASON_TEXTS:
        pattern_label = ", ".join(result.matched_patterns[:2]) if result.matched_patterns else "감지된 규칙"
        adjusted_reason = (
            f"{pattern_label} 관련 표현이 감지되어 정책상 최소 {minimum_level.value} 단계를 유지합니다."
        )
    else:
        adjusted_reason = result.reason
        if "정책상 최소" not in adjusted_reason:
            adjusted_reason = f"{adjusted_reason} 정책상 최소 {minimum_level.value} 단계를 유지합니다."

    return SentenceResult(
        sentence=result.sentence,
        suspicion_level=minimum_level,
        matched_keywords=result.matched_keywords,
        matched_patterns=result.matched_patterns,
        reason=adjusted_reason,
        score=max(result.score, minimum_score),
    )

MODEL_PATH = os.getenv("KOBERT_MODEL_PATH", "models/kobert_ad_classifier_relabel")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
_model_lock = threading.Lock()
ALLOWLIST_NORMAL_REASONS = {
    STRICT_ALLOW_NORMAL_REASON,
    EXTRA_ALLOW_NORMAL_REASON,
}
NORMAL_REASON_TEXTS = {
    DEFAULT_NORMAL_REASON,
    CONTEXT_ALLOW_NORMAL_REASON,
    NON_DOMAIN_NORMAL_REASON,
    NON_DOMAIN_IMAGE_NORMAL_REASON,
    UNCERTAIN_DOMAIN_REASON,
    "현재 기준에서는 문제 표현이 확인되지 않았습니다.",
    "짧은 광고 문구로 보이지만, 현재 기준에서는 문제 표현이 확인되지 않았습니다.",
}


def _infer_patterns_from_keywords(keywords: list[str]) -> list[str]:
    inferred: list[str] = []
    keyword_set = set(keywords)
    for pattern, pattern_keywords in FORBIDDEN_KEYWORDS.items():
        if keyword_set.intersection(pattern_keywords):
            inferred.append(pattern)
    return inferred


def _ensure_consistent_reason(result: SentenceResult) -> SentenceResult:
    if result.suspicion_level == SuspicionLevel.NORMAL:
        return result

    repaired_patterns = list(result.matched_patterns)
    if not repaired_patterns and result.matched_keywords:
        repaired_patterns = _infer_patterns_from_keywords(result.matched_keywords)

    repaired_base = SentenceResult(
        sentence=result.sentence,
        suspicion_level=result.suspicion_level,
        matched_keywords=result.matched_keywords,
        matched_patterns=repaired_patterns,
        reason=result.reason,
        score=result.score,
    )

    if repaired_base.reason not in NORMAL_REASON_TEXTS and repaired_base.matched_patterns == result.matched_patterns:
        return repaired_base

    repaired_reason = _build_kobert_reason(
        repaired_base.suspicion_level,
        repaired_base,
        repaired_base.score,
    )
    return SentenceResult(
        sentence=repaired_base.sentence,
        suspicion_level=repaired_base.suspicion_level,
        matched_keywords=repaired_base.matched_keywords,
        matched_patterns=repaired_base.matched_patterns,
        reason=repaired_reason,
        score=repaired_base.score,
    )


def _load_model():
    """KoBERT 모델 로드 (최초 1회만 실행)"""
    global _tokenizer, _model
    if _model is not None:
        return True
    with _model_lock:
        if _model is not None:
            return True
        prepare_model_runtime()
        if not os.path.exists(MODEL_PATH):
            print(f"[KoBERT] 모델 경로 없음: {MODEL_PATH}")
            return False
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
            _model.eval()
            return True
        except Exception as e:
            print(f"[KoBERT] 모델 로드 실패: {e}")
            return False


def is_kobert_loaded() -> bool:
    return _model is not None


def is_kobert_available() -> bool:
    return os.path.exists(MODEL_PATH)


def warm_kobert_model() -> bool:
    return _load_model()


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
        # The API normally skips these earlier, but keep this guard so direct
        # callers do not accidentally let KoBERT override obvious allowlisted text.
        if (
            rule_result.suspicion_level == SuspicionLevel.NORMAL
            and rule_result.reason in ALLOWLIST_NORMAL_REASONS
            and not rule_result.matched_patterns
        ):
            return rule_result

        # ── 짧은 문장 과탐지 방지 ─────────────────────────────
        # 인사말/단어 수준(글자수<6 또는 단어≤2)이며 규칙 패턴이 없으면 바로 정상 처리
        if (
            (len(sentence.strip()) < 6 or len(sentence.split()) <= 2)
            and not rule_result.matched_patterns
            and rule_result.reason != UNCERTAIN_DOMAIN_REASON
        ):
            return SentenceResult(
                sentence=sentence,
                suspicion_level=SuspicionLevel.NORMAL,
                matched_keywords=rule_result.matched_keywords,
                matched_patterns=rule_result.matched_patterns,
                reason="아주 짧은 일반 문구는 과탐지를 막기 위해 정상으로 처리했습니다.",
                score=0.0,
            )

        inputs = _tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        # Some saved KoBERT bundles expose an XLNet-style tokenizer that can emit
        # token_type_ids outside BERT's supported 0/1 range. The classifier does
        # not rely on segment IDs for single-sentence inference, so drop them.
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]  # [정상, 주의, 의심] 확률

        prob_normal     = probs[0].item()
        prob_caution    = probs[1].item()
        prob_suspicious = probs[2].item()

        # ── 확률 가중 점수 계산 ───────────────────────────────
        # 기본값을 낮춰 과탐지 완화: 주의는 0.3, 의심은 1.0
        weighted_score = (prob_caution * 0.3) + (prob_suspicious * 1.0)

        # 룰 패턴이 하나도 없으면 보수적으로 0.6배 클램프
        if not rule_result.matched_patterns:
            weighted_score *= 0.6

        # ── 가중 점수 → 의심도 레벨 변환 ────────────────────
        kobert_level = _resolve_kobert_level(rule_result, weighted_score)

        if (
            kobert_level == SuspicionLevel.SUSPICIOUS
            and rule_result.matched_patterns
            and set(rule_result.matched_patterns).issubset(CAUTION_ONLY_PATTERNS)
        ):
            kobert_level = SuspicionLevel.CAUTION

        # ── 최종 결과 반환 ────────────────────────────────────
        # KoBERT 결과가 규칙기반과 다르면 KoBERT 결과 채택 + 이유 생성
        # KoBERT 결과가 같으면 규칙기반 이유 그대로 유지
        if kobert_level != rule_result.suspicion_level:
            reason = _build_kobert_reason(kobert_level, rule_result, weighted_score)
            print(f"[KoBERT] 룰엔진:{rule_result.suspicion_level.name} -> KoBERT:{kobert_level.name} ({weighted_score:.2f})")
        else:
            reason = rule_result.reason

        # 정상으로 확정되면 점수를 0으로 리셋해 전체 점수에 영향 없도록
        final_score = 0.0 if kobert_level == SuspicionLevel.NORMAL else round(weighted_score, 3)

        kobert_result = SentenceResult(
            sentence=sentence,
            suspicion_level=kobert_level,
            matched_keywords=rule_result.matched_keywords,
            matched_patterns=rule_result.matched_patterns,
            reason=reason,
            score=final_score,
        )

        return _ensure_consistent_reason(_enforce_minimum_pattern_level(kobert_result))

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
        f"주요 문구: \"{top.sentence}\" - {top.reason} "
        f"광고 내용을 비판적으로 검토하시기 바랍니다."
    )
