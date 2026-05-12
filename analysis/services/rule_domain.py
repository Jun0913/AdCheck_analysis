COSMETIC_DOMAIN_NOUNS: frozenset[str] = frozenset({
    "크림", "로션", "세럼", "토너", "에센스", "스킨", "미스트",
    "선크림", "클렌저", "앰플", "오일", "스틱",
    "마스크팩", "시트마스크", "밤", "아이크림", "립밤", "립크림",
    "파운데이션", "비비크림", "쿠션", "컨실러", "파우더",
    "샴푸", "컨디셔너", "헤어토닉", "헤어에센스", "헤어오일",
    "바디로션", "바디크림", "바디워시",
    "화장품", "뷰티", "스킨케어", "기초화장품", "색조화장품",
    "메이크업", "코스메틱", "화장",
    "피부", "두피", "모발", "모공", "주름", "미백", "보습",
    "탄력", "트러블", "피부결", "피부장벽", "장벽", "각질", "탈모", "리프팅",
    "성분", "제형", "오일", "원료", "추출물", "향료",
    "자외선차단", "기능성화장품", "잡티",
})

COSMETIC_COPY_HINTS: frozenset[str] = frozenset({
    "수분",
    "속수분",
    "겉수분",
    "흡수",
    "스며들",
})


def is_cosmetic_related(sentence: str) -> bool:
    return any(noun in sentence for noun in COSMETIC_DOMAIN_NOUNS) or any(
        hint in sentence for hint in COSMETIC_COPY_HINTS
    )


def has_image_cosmetic_context(
    sentences: list[str],
    cosmetic_predictions: list[tuple[bool, float] | None],
) -> bool:
    return any(
        (pred is not None and pred[0]) or is_cosmetic_related(sentence)
        for pred, sentence in zip(cosmetic_predictions, sentences)
    )
