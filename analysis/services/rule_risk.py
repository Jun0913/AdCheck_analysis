import re


_SEMANTIC_PARTICLE_SUFFIXES = (
    "으로",
    "에게",
    "에서",
    "보다",
    "까지",
    "부터",
    "처럼",
    "만큼",
    "의",
    "을",
    "를",
    "이",
    "가",
    "은",
    "는",
    "에",
    "와",
    "과",
    "도",
    "로",
)


def _normalize_semantic_text(text: str) -> str:
    tokens = re.findall(r"[가-힣A-Za-z0-9%]+", text)
    normalized_tokens: list[str] = []

    for token in tokens:
        normalized = token
        if re.fullmatch(r"[가-힣]+", token):
            for suffix in _SEMANTIC_PARTICLE_SUFFIXES:
                if len(token) > len(suffix) + 1 and token.endswith(suffix):
                    normalized = token[: -len(suffix)]
                    break
        normalized_tokens.append(normalized)

    return "".join(normalized_tokens)


def detect_strong_forbidden_keyword(
    sentence: str,
    strong_forbidden_keywords: list[str],
) -> str | None:
    for keyword in strong_forbidden_keywords:
        if keyword in sentence:
            return keyword
    return None


def detect_forbidden_patterns(
    sentence: str,
    forbidden_keywords: dict[str, list[str]],
    forbidden_regex_patterns: dict[str, list[tuple[str, str]]],
) -> tuple[list[str], list[str]]:
    matched_keywords: list[str] = []
    matched_patterns: list[str] = []
    normalized_sentence = _normalize_semantic_text(sentence)

    for pattern_tag, keywords in forbidden_keywords.items():
        for keyword in keywords:
            if (
                keyword in sentence
                or _normalize_semantic_text(keyword) in normalized_sentence
            ):
                matched_keywords.append(keyword)
                if pattern_tag not in matched_patterns:
                    matched_patterns.append(pattern_tag)

    for pattern_tag, regex_items in forbidden_regex_patterns.items():
        for pattern, label in regex_items:
            if re.search(pattern, sentence):
                matched_keywords.append(label)
                if pattern_tag not in matched_patterns:
                    matched_patterns.append(pattern_tag)

    return matched_keywords, matched_patterns


def detect_high_risk_functional_claims(
    sentence: str,
    high_risk_functional_keywords: list[str],
) -> list[str]:
    normalized_sentence = _normalize_semantic_text(sentence)
    return [
        keyword
        for keyword in high_risk_functional_keywords
        if keyword in sentence or _normalize_semantic_text(keyword) in normalized_sentence
    ]


def detect_high_risk_functional_regex_matches(
    sentence: str,
    regex_patterns: list[tuple[str, str]],
) -> list[str]:
    return [
        label
        for pattern, label in regex_patterns
        if re.search(pattern, sentence)
    ]


def detect_high_risk_functional_combinations(
    sentence: str,
    base_keywords: list[str],
    context_keywords: list[str],
) -> list[str]:
    matched_bases = [keyword for keyword in base_keywords if keyword in sentence]
    if not matched_bases:
        return []

    matched_contexts = [keyword for keyword in context_keywords if keyword in sentence]
    if not matched_contexts:
        return []

    return matched_bases
