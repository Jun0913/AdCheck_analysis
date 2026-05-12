import re


def _normalize_allow_residual_token(token: str) -> str:
    normalized = re.sub(r"(이|가|은|는|을|를|에|의|와|과|도|로|으로|만|부터|까지)$", "", token)
    return normalized


def find_whitelist_matches(
    sentence: str,
    strict_allow_keywords: list[str],
    strict_allow_extra_keywords: list[str],
) -> list[str]:
    matches = [kw for kw in strict_allow_keywords if kw in sentence]
    matches.extend(kw for kw in strict_allow_extra_keywords if kw in sentence)
    return matches


def find_context_allow_matches(
    sentence: str,
    context_allow_keywords: list[str],
) -> list[str]:
    return [kw for kw in context_allow_keywords if kw in sentence]


def find_regex_allow_matches(
    sentence: str,
    allow_regex_patterns: list[str],
) -> list[str]:
    matches: list[str] = []
    for pattern in allow_regex_patterns:
        found = re.search(pattern, sentence)
        if found:
            matches.append(found.group(0))
    return matches


def is_clearly_allowed_sentence(
    sentence: str,
    allow_matches: list[str],
    allow_context_stopwords: set[str],
) -> bool:
    residual = sentence
    for kw in sorted(set(allow_matches), key=len, reverse=True):
        residual = residual.replace(kw, " ")

    normalized_stopwords = {
        _normalize_allow_residual_token(token)
        for token in allow_context_stopwords
    }
    tokens = re.findall(r"[가-힣A-Za-z0-9%+/]+", residual)
    meaningful_tokens = [
        token
        for token in (_normalize_allow_residual_token(token) for token in tokens)
        if len(token) >= 2 and token not in normalized_stopwords
    ]
    return len(meaningful_tokens) <= 1
