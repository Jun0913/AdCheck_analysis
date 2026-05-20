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

_GENERIC_METRIC_STOPWORDS = {
    "사용", "후", "전", "주", "일", "회", "직후", "만으로", "단", "기준", "결과",
}
_GENERIC_METRIC_PREFIX_RE = re.compile(
    r"^(?:사용\s*\d+\s*주\s*후|사용\s*\d+\s*일\s*후|사용\s*후|사용\s*직후|직후|"
    r"단\s*\d+\s*회\s*만으로|단\s*1회\s*만으로|1회\s*사용으로|인체적용시험\s*결과)\s*"
)


def _refine_generic_metric_label(
    sentence: str,
    match: re.Match[str],
    raw_metric: str,
) -> str:
    metric = " ".join(raw_metric.split()).strip()
    if len(metric) >= 2:
        return metric

    left_context = sentence[match.start(): match.start("percent")].strip()
    left_context = _GENERIC_METRIC_PREFIX_RE.sub("", left_context)
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", left_context)
    tokens = [
        token
        for token in tokens
        if token not in _GENERIC_METRIC_STOPWORDS
        and not re.fullmatch(r"\d+", token)
    ]
    if not tokens:
        return metric

    return " ".join(tokens[-3:]).strip() or metric


def detect_quantitative_claim(sentence: str) -> tuple[list[str], list[str]]:
    matched_keywords: list[str] = []
    matched_patterns: list[str] = []

    for idx, pattern in enumerate(_QUANT_PATTERNS):
        for match in pattern.finditer(sentence):
            metric = match.group("metric").strip()
            change = match.group("change").strip()
            percent = match.group("percent").strip()
            if idx >= 2:
                metric = _refine_generic_metric_label(sentence, match, metric)
            if len(metric) < 2:
                continue
            label = f"{metric} {percent} {change}"
            if label not in matched_keywords:
                matched_keywords.append(label)
        if matched_keywords and "효능과장" not in matched_patterns:
            matched_patterns.append("효능과장")

    return matched_keywords, matched_patterns
