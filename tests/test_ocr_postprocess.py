from analysis.services.extractor import _normalize_ocr_text


def test_ocr_normalize_fixes_common_cosmetic_terms():
    text = "여드름성 피부 사용 적합(논코메도제나)\n인체적용시힘 완료"
    normalized = _normalize_ocr_text(text)
    assert "논코메도제닉" in normalized
    assert "인체적용시험 완료" in normalized


def test_ocr_normalize_fixes_numeric_mm_pattern():
    text = "피부 광채(윤기), 결(매끄러움), 수분 레이어드(0.5mm/15mm)"
    normalized = _normalize_ocr_text(text)
    assert "0.5mm/1.5mm" in normalized
