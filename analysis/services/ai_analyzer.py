"""
2차 KoBERT 기반 분석 모듈
- 학습된 KoBERT 모델을 로드하여 문장별 의심도를 분류한다.
- 모델이 없을 경우 1차 규칙기반 결과를 그대로 반환한다.
"""

import os
from analysis.services.model_runtime import prepare_model_runtime

prepare_model_runtime()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analysis.models.schemas import SentenceResult, SuspicionLevel
from analysis.services.nli_verifier import verify_with_nli

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

MIN_PATTERN_LEVELS: dict[str, SuspicionLevel] = {
    "강력금지": SuspicionLevel.SUSPICIOUS,
    "의약품오인": SuspicionLevel.SUSPICIOUS,
    "효능과장": SuspicionLevel.SUSPICIOUS,
    "안전성단정": SuspicionLevel.SUSPICIOUS,
    "추천보증": SuspicionLevel.SUSPICIOUS,
    "검증오인": SuspicionLevel.SUSPICIOUS,
    "기능성오인": SuspicionLevel.CAUTION,
    "첨단기술오인": SuspicionLevel.CAUTION,
    "비교우위": SuspicionLevel.CAUTION,
}

# 패턴 태그 → 한국어 설명 (rule_engine과 동일하게 유지)
_PATTERN_DESC: dict[str, str] = {
    "의약품오인":  "의약품으로 오인될 수 있는 표현",
    "효능과장":   "효능을 과장하거나 절대적으로 단정하는 표현",
    "기능성오인":  "기능성 화장품 심사 없이 기능성을 주장하는 표현",
    "안전성단정":  "안전성을 근거 없이 단정하는 표현",
    "추천보증":   "전문가 추천·인증을 주장하는 표현",
    "검증오인":   "임상·시험 결과를 효능 보장처럼 단정하는 표현",
    "비교우위":   "근거 없이 타사 대비 우위를 주장하는 표현",
}

CAUTION_ONLY_PATTERNS = {"기능성오인", "첨단기술오인", "비교우위"}

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


def _enforce_minimum_pattern_level(result: SentenceResult) -> SentenceResult:
    minimum_level = SuspicionLevel.NORMAL
    for pattern in result.matched_patterns:
        pattern_level = MIN_PATTERN_LEVELS.get(pattern, SuspicionLevel.NORMAL)
        if _LEVEL_ORDER[pattern_level] > _LEVEL_ORDER[minimum_level]:
            minimum_level = pattern_level

    if _LEVEL_ORDER[result.suspicion_level] >= _LEVEL_ORDER[minimum_level]:
        return result

    minimum_score = {
        SuspicionLevel.CAUTION: 0.35,
        SuspicionLevel.SUSPICIOUS: 0.70,
    }.get(minimum_level, result.score)
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

MODEL_PATH = os.getenv("KOBERT_MODEL_PATH", "models/kobert_ad_classifier")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
WHITELIST_NORMAL_REASON = "허용된 표현 중심의 문구로 판단됩니다."


def _load_model():
    """KoBERT 모델 로드 (최초 1회만 실행)"""
    global _tokenizer, _model
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
            and rule_result.reason == WHITELIST_NORMAL_REASON
            and not rule_result.matched_patterns
        ):
            return rule_result

        # ── 짧은 문장 과탐지 방지 ─────────────────────────────
        # 인사말/단어 수준(글자수<6 또는 단어≤2)이며 규칙 패턴이 없으면 바로 정상 처리
        if (len(sentence.strip()) < 6 or len(sentence.split()) <= 2) and not rule_result.matched_patterns:
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
        # 규칙기반 결과에 따라 임계값 차등 적용
        # - 규칙기반이 의심: KoBERT가 확인/하향 조정
        # - 규칙기반이 주의: KoBERT가 올리거나 내릴 수 있음
        # - 규칙기반이 정상: KoBERT 임계값 높여서 과탐지 방지
        if rule_result.suspicion_level == SuspicionLevel.SUSPICIOUS:
            # 규칙기반이 의심 → KoBERT가 확인/하향 조정
            if weighted_score >= 0.55:
                kobert_level = SuspicionLevel.SUSPICIOUS
            elif weighted_score >= 0.30:
                kobert_level = SuspicionLevel.CAUTION
            else:
                kobert_level = SuspicionLevel.CAUTION  # 최소 주의 유지
        elif rule_result.suspicion_level == SuspicionLevel.CAUTION:
            # 규칙기반이 주의 → 올릴 수도 내릴 수도 있음 (보수적)
            if weighted_score >= 0.60:
                kobert_level = SuspicionLevel.SUSPICIOUS
            elif weighted_score >= 0.30:
                kobert_level = SuspicionLevel.CAUTION
            else:
                kobert_level = SuspicionLevel.NORMAL
        else:
            # 규칙기반이 정상 → 임계값을 더 높여 과탐지 방지
            if weighted_score >= 0.80:
                kobert_level = SuspicionLevel.SUSPICIOUS
            elif weighted_score >= 0.55:
                kobert_level = SuspicionLevel.CAUTION
            else:
                kobert_level = SuspicionLevel.NORMAL

        if (
            rule_result.suspicion_level == SuspicionLevel.CAUTION
            and rule_result.matched_keywords
            and kobert_level == SuspicionLevel.NORMAL
        ):
            kobert_level = SuspicionLevel.CAUTION

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

        # ── 3차: NLI 검증 ──────────────────────────────────────
        # 조건: 규칙 엔진이 패턴을 감지했고 KoBERT 점수가 회색지대(0.35~0.65)이거나
        #       규칙 엔진·KoBERT 결과가 엇갈릴 때
        should_call_nli = bool(rule_result.matched_patterns) and (
            0.35 <= weighted_score <= 0.65
            or rule_result.suspicion_level != kobert_level
        )
        if should_call_nli:
            return _enforce_minimum_pattern_level(
                verify_with_nli(sentence, kobert_result, rule_result.matched_patterns)
            )

        return _enforce_minimum_pattern_level(kobert_result)

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
