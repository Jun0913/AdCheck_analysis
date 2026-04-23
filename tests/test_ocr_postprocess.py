from analysis.services.extractor import _normalize_ocr_text, split_sentences


def test_ocr_normalize_fixes_common_cosmetic_terms():
    text = "여드름성 피부 사용 적합(논코메도제나)\n인체적용시힘 완료"
    normalized = _normalize_ocr_text(text)
    assert "논코메도제닉" in normalized
    assert "인체적용시험 완료" in normalized


def test_ocr_normalize_fixes_common_ad_copy_errors():
    text = "기미잡리\n탄력저하\n아무리 쉬어도 지쳐보이틀 피부\nCEQ로리셋하세오\n쎄럼 크림입나다"
    normalized = _normalize_ocr_text(text)
    assert "기미잡티" in normalized
    assert "탄력 저하" in normalized
    assert "지쳐 보이는 피부" in normalized
    assert "CEQ로 리셋하세요" in normalized
    assert "세럼 크림입니다" in normalized


def test_ocr_normalize_fixes_numeric_mm_pattern():
    text = "피부 광채(윤기), 결(매끄러움), 수분 레이어드(0.5mm/15mm)"
    normalized = _normalize_ocr_text(text)
    assert "0.5mm/1.5mm" in normalized


def test_split_sentences_merges_short_ocr_ad_fragments():
    text = "기미잡티\n탄력저하\n아무리 쉬어도 지쳐 보이는 피부\nCEQ로리셋하세요"
    sentences = split_sentences(text)
    assert sentences == ["기미잡티 탄력저하 아무리 쉬어도 지쳐 보이는 피부 CEQ로리셋하세요"]


def test_split_sentences_keeps_complete_sentences_separate():
    text = "이 크림은 아토피를 치료합니다.\n피부 보습에 도움을 주는 크림입니다."
    sentences = split_sentences(text)
    assert sentences == ["이 크림은 아토피를 치료합니다.", "피부 보습에 도움을 주는 크림입니다."]
