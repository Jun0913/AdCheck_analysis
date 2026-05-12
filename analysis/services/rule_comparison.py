COMPARATIVE_SUPERIORITY_KEYWORDS: list[str] = [
    "타사 대비", "경쟁사보다", "시중 제품보다",
    "기존 제품보다 월등", "압도적으로 뛰어난",
    "가장 효과적인", "가장 빠른", "가장 강력한",
]

COMPARATIVE_SUPERIORITY_REGEX_PATTERNS: list[tuple[str, str]] = [
    (r"(국내|업계|세계|전국)\s*(1위|최초)", "1위/최초"),
    (r"\b(BEST|Best|best)\b", "BEST"),
    (r"\bNo\.?\s*1\b", "No.1"),
    (r"(타사|기존\s*제품|경쟁사)\s*대비\s*\d+\s*배", "대비 2배"),
    (r"(기존\s*제품|타사\s*제품|타사)\s*보다\s*(더\s*)?(우수|뛰어난|좋은|강력한)", "타사 대비 우수"),
]
