from analysis.services.rule_allow import (
    find_context_allow_matches,
    find_regex_allow_matches,
    find_whitelist_matches,
    is_clearly_allowed_sentence,
)
from analysis.services.rule_comparison import (
    COMPARATIVE_SUPERIORITY_KEYWORDS,
    COMPARATIVE_SUPERIORITY_REGEX_PATTERNS,
)
from analysis.services.rule_caution import (
    CAUTION_KEYWORDS,
    CAUTION_REGEX_PATTERNS,
    detect_caution_matches,
)
from analysis.services.rule_domain import has_image_cosmetic_context, is_cosmetic_related
from analysis.services.rule_endorsement import (
    ENDORSEMENT_KEYWORDS,
    ENDORSEMENT_REGEX_PATTERNS,
)
from analysis.services.rule_functional import (
    ADVANCED_TECH_MISLEADING_KEYWORDS,
    EXAGGERATED_EFFICACY_KEYWORDS,
    EXAGGERATED_EFFICACY_REGEX_PATTERNS,
    FUNCTIONAL_MISLEADING_KEYWORDS,
    FUNCTIONAL_MISLEADING_REGEX_PATTERNS,
    HIGH_RISK_FUNCTIONAL_KEYWORDS,
)
from analysis.services.rule_medical import MEDICAL_MISLEADING_KEYWORDS
from analysis.services.rule_patterns import (
    CAUTION_ONLY_PATTERNS,
    MIN_PATTERN_LEVELS,
    PATTERN_REASON_TEMPLATES,
    STRONG_SUSPICIOUS_PATTERNS,
)
from analysis.services.rule_policy import (
    ALLOW_CONTEXT_STOPWORDS,
    CONTEXT_ALLOW_KEYWORDS,
    DOMAIN_ONLY_TERMS,
    EXACT_ALLOW_SENTENCES,
    EXACT_CAUTION_SENTENCES,
    STRICT_ALLOW_REGEX_PATTERNS,
    STRICT_ALLOW_EXTRA_KEYWORDS,
    STRICT_ALLOW_KEYWORDS,
)
from analysis.services.rule_risk import (
    detect_forbidden_patterns,
    detect_high_risk_functional_claims,
    detect_high_risk_functional_regex_matches,
    _normalize_semantic_text,
    detect_strong_forbidden_keyword,
)
from analysis.services.rule_safety import SAFETY_CLAIM_KEYWORDS
from analysis.services.rule_verification import VERIFICATION_MISLEADING_REGEX_PATTERNS


def test_is_cosmetic_related_accepts_short_cosmetic_copy() -> None:
    assert is_cosmetic_related("가볍게 쏙 스며들어 깊게 채워지는 첫 수분")


def test_has_image_cosmetic_context_uses_prediction_or_rule_signal() -> None:
    assert has_image_cosmetic_context(
        ["깊게 채워지는 첫 수분", "브랜드 로고"],
        [None, None],
    )
    assert has_image_cosmetic_context(
        ["텍스트 없음", "브랜드 로고"],
        [(True, 0.92), None],
    )


def test_allow_helpers_find_whitelist_and_context_matches() -> None:
    whitelist = find_whitelist_matches(
        "기능성화장품(미백에 도움을 줌)",
        ["기능성화장품(미백에 도움을 줌)"],
        ["미백에 도움을 줌"],
    )
    context = find_context_allow_matches(
        "자외선 차단 지수 SPF 50+",
        ["SPF", "PA"],
    )
    assert whitelist == ["기능성화장품(미백에 도움을 줌)", "미백에 도움을 줌"]
    assert context == ["SPF"]


def test_allow_helpers_find_regex_allow_matches() -> None:
    matches = find_regex_allow_matches(
        "집중 케어 여드름성 피부 사용 적합 인체 적용 시험 완료",
        STRICT_ALLOW_REGEX_PATTERNS,
    )
    assert "여드름성 피부 사용 적합" in matches


def test_is_clearly_allowed_sentence_rejects_mixed_claims() -> None:
    assert is_clearly_allowed_sentence(
        "기능성화장품(미백에 도움을 줌)",
        ["기능성화장품(미백에 도움을 줌)"],
        {"도움", "줌"},
    )
    assert not is_clearly_allowed_sentence(
        "기능성화장품(미백에 도움을 줌) 주름 88% 개선",
        ["기능성화장품(미백에 도움을 줌)"],
        {"도움", "줌"},
    )


def test_risk_helpers_detect_keyword_regex_and_high_risk_matches() -> None:
    matched_keywords, matched_patterns = detect_forbidden_patterns(
        "전문의 추천 제품",
        {"추천보증": ["병원 추천"]},
        {"추천보증": [(r"전문의\s*추천", "전문가 추천")]},
    )
    assert matched_keywords == ["전문가 추천"]
    assert matched_patterns == ["추천보증"]

    assert detect_high_risk_functional_claims(
        "DNA 활성화 성분",
        ["DNA 활성화", "줄기세포"],
    ) == ["DNA 활성화"]

    assert detect_strong_forbidden_keyword(
        "100% 개선 보장",
        ["100%", "평생 보장"],
    ) == "100%"

    assert detect_high_risk_functional_regex_matches(
        "속눈썹, 눈썹이 자람",
        [(r"(속눈썹|눈썹)[,\s]*(?:및\s*)?(속눈썹|눈썹)?[가-힣\s]*자람", "속눈썹, 눈썹이 자람")],
    ) == ["속눈썹, 눈썹이 자람"]


def test_semantic_normalization_ignores_spacing_and_particles_for_keywords() -> None:
    assert _normalize_semantic_text("모발의 두께를 증가") == _normalize_semantic_text("모발 두께 증가")
    assert _normalize_semantic_text("유익균의 균형 보호") == _normalize_semantic_text("유익균 균형 보호")

    matched_keywords, matched_patterns = detect_forbidden_patterns(
        "유익균의 균형 보호",
        {"기능성오인": ["유익균 균형 보호"]},
        {},
    )
    assert matched_keywords == ["유익균 균형 보호"]
    assert matched_patterns == ["기능성오인"]


def test_pattern_metadata_keeps_shared_policy_boundaries() -> None:
    assert MIN_PATTERN_LEVELS["강력금지"].value == "의심"
    assert "주의예외" in CAUTION_ONLY_PATTERNS
    assert "추천보증" in STRONG_SUSPICIOUS_PATTERNS
    assert "치료나 예방 효과로 받아들여질 수 있는 표현" in PATTERN_REASON_TEMPLATES["의약품오인"]


def test_split_risk_category_dictionaries_keep_expected_entries() -> None:
    assert "치료" in MEDICAL_MISLEADING_KEYWORDS
    assert "의사 추천" in ENDORSEMENT_KEYWORDS
    assert any(label == "전문가 추천" for _, label in ENDORSEMENT_REGEX_PATTERNS)
    assert "타사 대비" in COMPARATIVE_SUPERIORITY_KEYWORDS
    assert any(label == "1위/최초" for _, label in COMPARATIVE_SUPERIORITY_REGEX_PATTERNS)


def test_split_functional_dictionaries_keep_expected_entries() -> None:
    assert "즉시 효과" in EXAGGERATED_EFFICACY_KEYWORDS
    assert any(label == "99.9%" for _, label in EXAGGERATED_EFFICACY_REGEX_PATTERNS)
    assert "모공 제거" in FUNCTIONAL_MISLEADING_KEYWORDS
    assert any(label == "기능성 효능" for _, label in FUNCTIONAL_MISLEADING_REGEX_PATTERNS)
    assert "줄기세포" in ADVANCED_TECH_MISLEADING_KEYWORDS
    assert "DNA 활성화" in HIGH_RISK_FUNCTIONAL_KEYWORDS


def test_split_safety_verification_and_caution_helpers_keep_expected_entries() -> None:
    assert "부작용 없음" in SAFETY_CLAIM_KEYWORDS
    assert any(label == "임상 입증" for _, label in VERIFICATION_MISLEADING_REGEX_PATTERNS)
    assert "24시간" in CAUTION_KEYWORDS
    assert any(label == "지속" for _, label in CAUTION_REGEX_PATTERNS)
    assert detect_caution_matches("24시간 지속되는 롱래스팅") == ["24시간", "롱래스팅", "24시간"]


def test_split_allow_policy_dictionaries_keep_expected_entries() -> None:
    assert "피부 보습에 도움" in STRICT_ALLOW_KEYWORDS
    assert "탈모 증상 완화에 도움을 주는 화장품" in STRICT_ALLOW_EXTRA_KEYWORDS
    assert any("여드름성" in pattern for pattern in STRICT_ALLOW_REGEX_PATTERNS)
    assert "주름개선 도움" in EXACT_ALLOW_SENTENCES
    assert "피부 가려움 완화" in EXACT_CAUTION_SENTENCES
    assert "피부과 테스트 완료" in CONTEXT_ALLOW_KEYWORDS
    assert "크림" in DOMAIN_ONLY_TERMS
    assert "도움" in ALLOW_CONTEXT_STOPWORDS
