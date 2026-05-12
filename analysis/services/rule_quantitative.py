import re


_METRIC_TERMS = (
    "유분량",
    "유분",
    "피지량",
    "피지",
    "겉피지",
    "속피지",
    "모공",
    "블랙헤드",
    "화이트헤드",
    "각질량",
    "각질",
    "피부결",
    "붉은기",
    "피부온도",
    "온도",
    "멜라닌",
    "면포",
    "노폐물",
    "수분",
    "수분감",
    "수분량",
    "수분 함량",
    "보습",
    "장벽",
    "수분 장벽",
    "피부 장벽",
)

_CHANGE_TERMS = ("감소", "개선", "하락", "억제", "완화", "증가", "상승")
_PERCENT_RE = r"-?\d+(?:\.\d+)?%"
_CONNECTOR_RE = r"[가-힣A-Za-z0-9\s()/,&+\-]{0,24}?"
_METRIC_RE = "|".join(sorted(_METRIC_TERMS, key=len, reverse=True))
_CHANGE_RE = "|".join(_CHANGE_TERMS)

_GENERIC_METRIC_TOKEN_RE = r"[가-힣A-Za-z][가-힣A-Za-z0-9\s]{0,16}?"

_QUANT_PATTERNS = (
    re.compile(
        rf"(?P<metric>{_METRIC_RE}){_CONNECTOR_RE}(?P<change>{_CHANGE_RE}){_CONNECTOR_RE}(?P<percent>{_PERCENT_RE})"
    ),
    re.compile(
        rf"(?P<metric>{_METRIC_RE}){_CONNECTOR_RE}(?P<percent>{_PERCENT_RE}){_CONNECTOR_RE}(?P<change>{_CHANGE_RE})"
    ),
    re.compile(
        rf"(?P<metric>{_GENERIC_METRIC_TOKEN_RE}){_CONNECTOR_RE}(?P<change>{_CHANGE_RE}){_CONNECTOR_RE}(?P<percent>{_PERCENT_RE})"
    ),
    re.compile(
        rf"(?P<metric>{_GENERIC_METRIC_TOKEN_RE}){_CONNECTOR_RE}(?P<percent>{_PERCENT_RE}){_CONNECTOR_RE}(?P<change>{_CHANGE_RE})"
    ),
)


def detect_quantitative_claim(sentence: str) -> tuple[list[str], list[str]]:
    matched_keywords: list[str] = []
    matched_patterns: list[str] = []

    for pattern in _QUANT_PATTERNS:
        for match in pattern.finditer(sentence):
            metric = match.group("metric").strip()
            change = match.group("change").strip()
            percent = match.group("percent").strip()
            label = f"{metric} {percent} {change}"
            if label not in matched_keywords:
                matched_keywords.append(label)
        if matched_keywords and "효능과장" not in matched_patterns:
            matched_patterns.append("효능과장")

    return matched_keywords, matched_patterns
