import re
from analysis.models.schemas import SuspicionLevel, SentenceResult
from analysis.services.rule_allow import (
    find_context_allow_matches as _find_context_allow_matches_impl,
    find_regex_allow_matches as _find_regex_allow_matches_impl,
    find_whitelist_matches as _find_whitelist_matches_impl,
    is_clearly_allowed_sentence as _is_clearly_allowed_sentence_impl,
)
from analysis.services.rule_domain import is_cosmetic_related as _is_cosmetic_related
from analysis.services.rule_comparison import (
    COMPARATIVE_SUPERIORITY_KEYWORDS,
    COMPARATIVE_SUPERIORITY_REGEX_PATTERNS,
)
from analysis.services.rule_caution import detect_caution_matches
from analysis.services.rule_endorsement import (
    ENDORSEMENT_KEYWORDS,
    ENDORSEMENT_REGEX_PATTERNS,
    REGULATORY_ALLOW_KEYWORDS,
    REGULATORY_ALLOW_REGEX_LABELS,
)
from analysis.services.rule_functional import (
    ADVANCED_TECH_MISLEADING_KEYWORDS,
    EXAGGERATED_EFFICACY_KEYWORDS,
    EXAGGERATED_EFFICACY_REGEX_PATTERNS,
    FUNCTIONAL_MISLEADING_KEYWORDS,
    FUNCTIONAL_MISLEADING_REGEX_PATTERNS,
    FUNCTIONAL_MISLEADING_VARIANT_REGEX_PATTERNS,
    HIGH_RISK_FUNCTIONAL_BASE_KEYWORDS,
    HIGH_RISK_FUNCTIONAL_CONTEXT_KEYWORDS,
    HIGH_RISK_FUNCTIONAL_KEYWORDS,
    HIGH_RISK_FUNCTIONAL_REGEX_PATTERNS,
)
from analysis.services.rule_medical import MEDICAL_MISLEADING_KEYWORDS
from analysis.services.rule_patterns import (
    PATTERN_REASON_TEMPLATES,
    PATTERN_RULE_LEVELS,
)
from analysis.services.rule_policy import (
    ALLOW_CONTEXT_STOPWORDS,
    CONTEXT_ALLOW_KEYWORDS,
    EXACT_ALLOW_SENTENCES,
    EXACT_CAUTION_SENTENCES,
    STRICT_ALLOW_REGEX_PATTERNS,
    STRICT_ALLOW_EXTRA_KEYWORDS,
    STRICT_ALLOW_KEYWORDS,
)
from analysis.services.rule_quantitative import detect_quantitative_claim
from analysis.services.rule_risk import (
    detect_forbidden_patterns as _detect_forbidden_patterns_impl,
    detect_high_risk_functional_combinations as _detect_high_risk_functional_combinations_impl,
    detect_high_risk_functional_claims as _detect_high_risk_functional_claims_impl,
    detect_high_risk_functional_regex_matches as _detect_high_risk_functional_regex_matches_impl,
    detect_strong_forbidden_keyword as _detect_strong_forbidden_keyword_impl,
)
from analysis.services.rule_safety import SAFETY_CLAIM_KEYWORDS
from analysis.services.rule_verification import VERIFICATION_MISLEADING_REGEX_PATTERNS

NON_DOMAIN_NORMAL_REASON = "화장품 광고 문맥이 확인되지 않았습니다."
NON_DOMAIN_IMAGE_NORMAL_REASON = "이미지 문구만으로는 화장품 광고 문맥을 확인하기 어려웠습니다."
UNCERTAIN_DOMAIN_REASON = "짧은 광고 문구로 보여 추가 확인이 필요한 표현입니다."
STRICT_ALLOW_NORMAL_REASON = "허용 범위의 표현으로 판단됩니다."
EXTRA_ALLOW_NORMAL_REASON = "허용 가능한 기능성 표현으로 판단됩니다."
CONTEXT_ALLOW_NORMAL_REASON = "안내 또는 일반 설명에 가까운 문구로 보입니다."
DEFAULT_NORMAL_REASON = "문제될 만한 표현은 확인되지 않았습니다."

# 화장품 여부와 무관하게 강하게 차단할 표현 (도메인 필터 우회)
STRONG_FORBIDDEN_KEYWORDS: list[str] = [
    "100%", "100퍼", "100퍼센트", "100프로",
    "영구", "영구제거", "평생 보장", "완치", "완벽", "즉시 효과",
    "의사 보증", "의사추천", "의사 인증",
    "부작용 없음", "전혀 부작용", "무조건 환불",
    "평생 유지", "확실한 효과",
]

# ────────────────────────────────────────────
# 금지 키워드 사전
# 출처: 식품의약품안전처 화장품 표시·광고 가이드라인,
#       공정거래위원회 표시광고법 위반 사례집
# ────────────────────────────────────────────

FORBIDDEN_KEYWORDS: dict[str, list[str]] = {

    # ── 의약품 오인 / 질병 치료·예방 표현 ──────────────────
    "의약품오인": MEDICAL_MISLEADING_KEYWORDS,

    # ── 효능 과장 / 절대적·단정적 표현 ────────────────────
    "효능과장": EXAGGERATED_EFFICACY_KEYWORDS,

    # ── 기능성 화장품 오인 (심사 없이 기능성 주장) ─────────
    "기능성오인": FUNCTIONAL_MISLEADING_KEYWORDS,

    # ── 안전성 단정 표현 ────────────────────────────────
    "안전성단정": SAFETY_CLAIM_KEYWORDS,

    # ── 추천·보증·인증 표현 ─────────────────────────────
    "추천보증": ENDORSEMENT_KEYWORDS,

    # ── 비교 우위 단정 표현 ─────────────────────────────
    "비교우위": COMPARATIVE_SUPERIORITY_KEYWORDS,

    # ── 첨단 기술·재생 과대 주장 (기능성 오인 세부) ─────────
    "첨단기술오인": ADVANCED_TECH_MISLEADING_KEYWORDS,
}

FORBIDDEN_REGEX_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "추천보증": ENDORSEMENT_REGEX_PATTERNS,
    "검증오인": VERIFICATION_MISLEADING_REGEX_PATTERNS,
    "비교우위": COMPARATIVE_SUPERIORITY_REGEX_PATTERNS,
    "효능과장": EXAGGERATED_EFFICACY_REGEX_PATTERNS,
    "기능성오인": FUNCTIONAL_MISLEADING_REGEX_PATTERNS + FUNCTIONAL_MISLEADING_VARIANT_REGEX_PATTERNS,
}

# ────────────────────────────────────────────
# 주의 키워드 (과장 가능성 있으나 단정 어려운 표현)
# ────────────────────────────────────────────

# ────────────────────────────────────────────
# 화장품 도메인 명사 (관련성 필터)
# 이 중 하나라도 포함돼야 분석 대상으로 처리
# 해당 명사가 없는 문장은 화장품 광고와 무관한 것으로 보고 정상 처리
# ────────────────────────────────────────────

COSMETIC_DOMAIN_NOUNS: frozenset[str] = frozenset({
    # 제품 종류
    "크림", "로션", "세럼", "앰플", "에센스", "토너", "스킨", "미스트",
    "선크림", "선스크린", "선블록", "클렌저", "클렌징", "폼클렌징",
    "마스크팩", "시트마스크", "팩", "아이크림", "립밤", "립크림",
    "파운데이션", "비비크림", "쿠션", "컨실러", "파우더",
    "샴푸", "컨디셔너", "트리트먼트", "헤어에센스", "헤어오일",
    "바디로션", "바디크림", "바디워시",
    # 화장품·뷰티 범주어
    "화장품", "뷰티", "스킨케어", "기초화장품", "색조화장품",
    "메이크업", "코스메틱", "화장",
    # 피부·두피·모발
    "피부", "두피", "모발", "모공", "주름", "미백", "보습",
    "탄력", "피부톤", "피부결", "피부장벽", "장벽", "각질", "탈모", "리프팅",
    # 성분·제형
    "성분", "제형", "함유", "원료", "추출물", "포뮬러",
    # 기능성
    "자외선차단", "기능성화장품",
    "잡티",
})



def is_cosmetic_related(sentence: str) -> bool:
    return _is_cosmetic_related(sentence)


def _find_forbidden_patterns(sentence: str) -> tuple[list[str], list[str]]:
    return _detect_forbidden_patterns_impl(
        sentence,
        FORBIDDEN_KEYWORDS,
        FORBIDDEN_REGEX_PATTERNS,
    )


def _find_high_risk_functional_matches(sentence: str) -> list[str]:
    return _detect_high_risk_functional_claims_impl(
        sentence,
        HIGH_RISK_FUNCTIONAL_KEYWORDS,
    )


def _find_high_risk_functional_combinations(sentence: str) -> list[str]:
    return _detect_high_risk_functional_combinations_impl(
        sentence,
        HIGH_RISK_FUNCTIONAL_BASE_KEYWORDS,
        HIGH_RISK_FUNCTIONAL_CONTEXT_KEYWORDS,
    )


def _find_high_risk_functional_regex_matches(sentence: str) -> list[str]:
    return _detect_high_risk_functional_regex_matches_impl(
        sentence,
        HIGH_RISK_FUNCTIONAL_REGEX_PATTERNS,
    )


def _find_strong_forbidden_keyword(sentence: str) -> str | None:
    return _detect_strong_forbidden_keyword_impl(
        sentence,
        STRONG_FORBIDDEN_KEYWORDS,
    )


def _find_whitelist_matches(sentence: str) -> list[str]:
    return _find_whitelist_matches_impl(
        sentence,
        STRICT_ALLOW_KEYWORDS,
        STRICT_ALLOW_EXTRA_KEYWORDS,
    )


def _has_only_relaxable_allow_patterns(matched_patterns: list[str]) -> bool:
    if not matched_patterns:
        return True
    relaxable_patterns = {"의약품오인", "기능성오인", "추천보증"}
    return set(matched_patterns).issubset(relaxable_patterns)


def _relax_regulatory_endorsement_matches(
    matched_keywords: list[str],
    matched_patterns: list[str],
    whitelist_matches: list[str],
    context_allow_matches: list[str],
) -> tuple[list[str], list[str]]:
    if (not whitelist_matches and not context_allow_matches) or "추천보증" not in matched_patterns:
        return matched_keywords, matched_patterns

    filtered_keywords = [
        keyword
        for keyword in matched_keywords
        if keyword not in REGULATORY_ALLOW_KEYWORDS
        and keyword not in REGULATORY_ALLOW_REGEX_LABELS
    ]

    removed_count = len(matched_keywords) - len(filtered_keywords)
    if removed_count == 0:
        return matched_keywords, matched_patterns

    remaining_endorsement = any(
        keyword in ENDORSEMENT_KEYWORDS or keyword in REGULATORY_ALLOW_REGEX_LABELS
        for keyword in filtered_keywords
    )
    if not remaining_endorsement:
        matched_patterns = [pattern for pattern in matched_patterns if pattern != "추천보증"]

    return filtered_keywords, matched_patterns


def _relax_caution_matches_for_allow_context(
    caution_matches: list[str],
    whitelist_matches: list[str],
    context_allow_matches: list[str],
) -> list[str]:
    if not whitelist_matches and not context_allow_matches:
        return caution_matches
    if not caution_matches:
        return caution_matches

    soft_caution = {"24시간", "지속"}
    context_relaxable_caution = {"해결"}
    if set(caution_matches).issubset(soft_caution):
        return []
    if context_allow_matches and set(caution_matches).issubset(context_relaxable_caution):
        return []
    return caution_matches


def _find_context_allow_matches(sentence: str) -> list[str]:
    return _find_context_allow_matches_impl(sentence, CONTEXT_ALLOW_KEYWORDS)


def _find_regex_allow_matches(sentence: str) -> list[str]:
    return _find_regex_allow_matches_impl(sentence, STRICT_ALLOW_REGEX_PATTERNS)


def _is_clearly_allowed_sentence(sentence: str, whitelist_matches: list[str]) -> bool:
    return _is_clearly_allowed_sentence_impl(
        sentence,
        whitelist_matches,
        ALLOW_CONTEXT_STOPWORDS,
    )


def _should_defer_non_domain_to_kobert(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False

    # Short ad-copy fragments with quantitative/ingredient-style tokens are often
    # ambiguous rather than clearly non-domain, so let KoBERT inspect them.
    if "%" in stripped or re.search(r"\d", stripped):
        return True

    tokens = re.findall(r"[가-힣A-Za-z0-9%]+", stripped)
    if len(stripped) <= 28 and any(len(token) >= 7 for token in tokens):
        return True

    # Short product/ingredient-style ad copy is often too ambiguous for the
    # non-domain filter to end confidently.
    if len(stripped) <= 32 and re.search(
        r"(함유|영양|세정|저자극|누적판매|제품(?:입니다)?|씻어내지|1위)",
        stripped,
    ):
        return True

    return False


def should_run_kobert(rule_result: SentenceResult) -> bool:
    """
    KoBERT는 규칙이 최종 확정한 정상 문장을 제외하고 최대한 태운다.
    - 비도메인/명백한 허용 표현은 스킵
    - 애매하지만 규칙상 정상인 문장은 KoBERT로 보낸다
    """
    if rule_result.suspicion_level != SuspicionLevel.NORMAL:
        return True

    if rule_result.reason in {
        NON_DOMAIN_NORMAL_REASON,
        STRICT_ALLOW_NORMAL_REASON,
        EXTRA_ALLOW_NORMAL_REASON,
    }:
        return False

    return True


# ────────────────────────────────────────────
# 패턴 태그 → 한국어 설명 매핑
# ────────────────────────────────────────────
# ────────────────────────────────────────────
# 분석 함수
# ────────────────────────────────────────────

def analyze_sentence(
    sentence: str,
    *,
    force_cosmetic: bool = False,
    source_type: str = "text",
) -> SentenceResult:
    """1차 규칙기반 엔진: 문장 하나를 분석하여 SentenceResult 반환"""
    normalized_sentence = re.sub(r"100\s*프로", "100프로", sentence)
    normalized_sentence = re.sub(r"100\s*퍼(?:센트)?", "100퍼센트", normalized_sentence)

    if sentence.strip() in EXACT_ALLOW_SENTENCES:
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=EXTRA_ALLOW_NORMAL_REASON,
            score=0.0,
        )

    if sentence.strip() in EXACT_CAUTION_SENTENCES:
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.CAUTION,
            matched_keywords=[],
            matched_patterns=["주의예외"],
            reason="허용 가능성이 있으나 효능 표현이 강해 주의가 필요한 문구입니다.",
            score=0.35,
        )

    # 0) 화장품 여부와 무관한 강력 금지 표현 우선 차단
    strong_forbidden_keyword = _find_strong_forbidden_keyword(normalized_sentence)
    if strong_forbidden_keyword:
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.SUSPICIOUS,
            matched_keywords=[strong_forbidden_keyword],
            matched_patterns=["강력금지"],
            reason=f"'{strong_forbidden_keyword}'처럼 허위/과장 소지가 큰 표현을 포함합니다.",
            score=0.85,
        )

    matched_keywords, matched_patterns = _find_forbidden_patterns(normalized_sentence)

    quantitative_keywords, quantitative_patterns = detect_quantitative_claim(normalized_sentence)
    matched_keywords.extend(quantitative_keywords)
    for pattern in quantitative_patterns:
        if pattern not in matched_patterns:
            matched_patterns.append(pattern)

    high_risk_matches = _find_high_risk_functional_matches(normalized_sentence)
    high_risk_matches.extend(
        match for match in _find_high_risk_functional_combinations(normalized_sentence)
        if match not in high_risk_matches
    )
    high_risk_matches.extend(
        match for match in _find_high_risk_functional_regex_matches(normalized_sentence)
        if match not in high_risk_matches
    )
    if high_risk_matches:
        matched_keywords.extend(high_risk_matches)
        if "고위험기능성오인" not in matched_patterns:
            matched_patterns.append("고위험기능성오인")

    if "무심사 기능성 표현" in matched_keywords:
        if "고위험기능성오인" not in matched_patterns:
            matched_patterns.append("고위험기능성오인")

    # 주의 키워드 검사
    caution_matches = detect_caution_matches(normalized_sentence)
    matched_keywords.extend(caution_matches)

    whitelist_matches = _find_whitelist_matches(normalized_sentence)
    context_allow_matches = _find_context_allow_matches(normalized_sentence)
    regex_allow_matches = _find_regex_allow_matches(normalized_sentence)
    extra_allow_matches = [kw for kw in STRICT_ALLOW_EXTRA_KEYWORDS if kw in normalized_sentence]
    matched_keywords, matched_patterns = _relax_regulatory_endorsement_matches(
        matched_keywords,
        matched_patterns,
        whitelist_matches + regex_allow_matches,
        context_allow_matches,
    )
    caution_matches = _relax_caution_matches_for_allow_context(
        caution_matches,
        whitelist_matches + regex_allow_matches,
        context_allow_matches,
    )
    combined_allow_matches = (
        whitelist_matches
        + extra_allow_matches
        + context_allow_matches
        + regex_allow_matches
    )

    # ── 도메인 관련성 필터 ─────────────────────────────────
    # 명백한 금지/주의 신호가 있으면 도메인 단서가 부족해도 계속 분석한다.
    if (
        not force_cosmetic
        and not is_cosmetic_related(normalized_sentence)
        and not matched_patterns
        and not caution_matches
        and not whitelist_matches
        and not context_allow_matches
        and not regex_allow_matches
    ):
        if _should_defer_non_domain_to_kobert(normalized_sentence):
            return SentenceResult(
                sentence=sentence,
                suspicion_level=SuspicionLevel.NORMAL,
                matched_keywords=[],
                matched_patterns=[],
                reason=UNCERTAIN_DOMAIN_REASON,
                score=0.0,
            )
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=(
                NON_DOMAIN_IMAGE_NORMAL_REASON
                if source_type == "image"
                else NON_DOMAIN_NORMAL_REASON
            ),
            score=0.0,
        )

    if (extra_allow_matches or regex_allow_matches) and _is_clearly_allowed_sentence(normalized_sentence, combined_allow_matches):
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=EXTRA_ALLOW_NORMAL_REASON,
            score=0.0,
        )

    # ── 화이트리스트 체크 ───────────────────────────────────
    # 금지 패턴이 없고, 문장 전체가 허용 맥락일 때만 정상 처리한다.
    if (
        whitelist_matches
        and _is_clearly_allowed_sentence(normalized_sentence, combined_allow_matches)
        and _has_only_relaxable_allow_patterns(matched_patterns)
    ):
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=STRICT_ALLOW_NORMAL_REASON,
            score=0.0,
        )

    if context_allow_matches and not matched_patterns and not caution_matches:
        matched_keywords.extend(context_allow_matches)

    # 의심도 결정 + 규칙 기반 연속 점수
    if matched_patterns:
        level = max(
            (PATTERN_RULE_LEVELS.get(pattern, SuspicionLevel.SUSPICIOUS) for pattern in matched_patterns),
            key=lambda lvl: {SuspicionLevel.NORMAL: 0, SuspicionLevel.CAUTION: 1, SuspicionLevel.SUSPICIOUS: 2}[lvl],
        )
        if level == SuspicionLevel.SUSPICIOUS:
            base_score = round(min(0.70 + len(matched_patterns) * 0.05, 0.90), 3)
        else:
            base_score = round(min(0.35 + len(matched_patterns) * 0.05, 0.55), 3)
        reason = _build_reason(matched_patterns, matched_keywords)
    elif caution_matches:
        level = SuspicionLevel.CAUTION
        # 주의 키워드 수에 따라 점수 차등 (0.35 ~ 0.55)
        base_score = round(min(0.35 + len(caution_matches) * 0.05, 0.55), 3)
        kw_display = ", ".join(f"'{kw}'" for kw in caution_matches[:3])
        reason = f"{kw_display} 표현은 효과를 과장하거나 오인을 유발할 가능성이 있어 주의가 필요합니다."
    else:
        level = SuspicionLevel.NORMAL
        base_score = 0.0
        reason = (
            CONTEXT_ALLOW_NORMAL_REASON
            if context_allow_matches
            else DEFAULT_NORMAL_REASON
        )

    return SentenceResult(
        sentence=sentence,
        suspicion_level=level,
        matched_keywords=list(set(matched_keywords)),
        matched_patterns=matched_patterns,
        reason=reason,
        score=base_score,
    )


def _build_reason(patterns: list[str], keywords: list[str]) -> str:
    """패턴 태그 + 감지 키워드를 조합해 문구별 맞춤 이유 생성"""
    if not patterns:
        return DEFAULT_NORMAL_REASON

    reasons = []
    used_keywords = set()

    for pattern in patterns:
        template = PATTERN_REASON_TEMPLATES.get(pattern)
        if not template:
            continue

        # 해당 패턴에 속하는 키워드 중 아직 사용 안 된 것 선택
        pattern_keywords = FORBIDDEN_KEYWORDS.get(pattern, [])
        matched = [kw for kw in keywords if kw in pattern_keywords and kw not in used_keywords]

        if matched:
            kw_display = ", ".join(f"'{kw}'" for kw in matched[:2])  # 최대 2개
            reason = template.replace("'{kw}'", kw_display)
            used_keywords.update(matched[:2])
        else:
            # 해당 패턴 키워드가 없으면 감지된 키워드 중 첫 번째 사용
            remaining = [kw for kw in keywords if kw not in used_keywords]
            if remaining:
                kw_display = f"'{remaining[0]}'"
                reason = template.replace("'{kw}'", kw_display)
                used_keywords.add(remaining[0])
            else:
                reason = template.replace("'{kw}' 표현은", "해당 표현은")

        reasons.append(reason)

    return " / ".join(reasons)


def calculate_overall_score(results: list[SentenceResult]) -> tuple[float, SuspicionLevel]:
    """전체 의심도 점수 및 레벨 계산 (연속 점수 기반 가중 평균)"""
    if not results:
        return 0.0, SuspicionLevel.NORMAL

    scores = sorted([r.score for r in results], reverse=True)
    n = len(scores)

    # 상위 1/3 문장에 가중치 2배 → 의심 문장이 전체 점수에 확실히 반영
    k = max(1, n // 3)
    weighted_sum = sum(scores[:k]) * 2.0 + sum(scores[k:])
    total_weight = k * 2.0 + (n - k)
    overall = weighted_sum / total_weight

    # 레벨 결정
    if overall >= 0.55:
        level = SuspicionLevel.SUSPICIOUS
    elif overall >= 0.20:
        level = SuspicionLevel.CAUTION
    else:
        level = SuspicionLevel.NORMAL

    # 고신뢰 의심 문장(score >= 0.80, 패턴 2개 이상)이 있으면 최소 주의 보장
    if any(r.suspicion_level == SuspicionLevel.SUSPICIOUS and r.score >= 0.80 for r in results):
        if level == SuspicionLevel.NORMAL:
            level = SuspicionLevel.CAUTION

    return round(overall, 3), level
