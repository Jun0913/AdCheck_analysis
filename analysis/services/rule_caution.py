import re


CAUTION_KEYWORDS: list[str] = [
    "24시간",
    "롱래스팅",
    "론웨어",
    "long lasting",
    "long-lasting",
    "해결",
    "피부나이",
    "동물 실험 없이",
    "동물 실험 X",
    "비건 인증",
    "기미잡티",
    "기미 잡티",
    "탄력 저하",
    "리셋",
]

CAUTION_REGEX_PATTERNS: list[tuple[str, str]] = [
    (r"24\s*시간", "24시간"),
    (r"(하루\s*종일|오래)\s*지속", "지속"),
    (r"(임상\s*테스트|인체\s*적용\s*시험)\s*(완료|완료됨|기반)", "임상 테스트"),
]


def detect_caution_matches(sentence: str) -> list[str]:
    matches = [keyword for keyword in CAUTION_KEYWORDS if keyword in sentence]
    matches.extend(
        label
        for pattern, label in CAUTION_REGEX_PATTERNS
        if re.search(pattern, sentence)
    )
    return matches
