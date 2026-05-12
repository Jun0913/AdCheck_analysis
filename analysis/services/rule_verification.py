VERIFICATION_MISLEADING_REGEX_PATTERNS: list[tuple[str, str]] = [
    (
        r"(임상(적으로)?|인체\s*적용\s*시험(으로|에서)?|테스트(로|에서)?)[가-힣A-Za-z0-9\s]{0,30}?(입증|검증|확인|증명|보장)",
        "임상 입증",
    ),
]
