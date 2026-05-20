import asyncio

from analysis.models.schemas import SuspicionLevel
from analysis.services.ai_analyzer import analyze_with_kobert
from analysis.services.rule_engine import analyze_sentence
from analysis.services.rule_quantitative import detect_quantitative_claim


def test_quantitative_rule_layer_detects_percent_change_claim():
    keywords, patterns = detect_quantitative_claim("피부결 감소 -87%")
    assert "효능과장" in patterns
    assert keywords


def test_quantitative_rule_layer_detects_generalized_moisture_metric_claim():
    keywords, patterns = detect_quantitative_claim("피부 수분 함량 232.45% 개선")
    assert "효능과장" in patterns
    assert keywords


def test_rule_engine_marks_percent_change_claim_as_caution():
    rule = analyze_sentence("피부결 감소 -87%", force_cosmetic=True)
    assert rule.suspicion_level == SuspicionLevel.CAUTION
    assert "효능과장" in rule.matched_patterns


def test_pipeline_keeps_minimum_caution_for_percent_change_claim():
    text = "피부결 감소 -87%"
    rule = analyze_sentence(text, force_cosmetic=True)
    final = asyncio.run(analyze_with_kobert(text, rule))
    assert final.suspicion_level == SuspicionLevel.CAUTION
    assert final.score >= 0.35


def test_pipeline_keeps_minimum_caution_for_moisture_percent_claim():
    text = "피부 수분 함량 232.45% 개선 수분 장벽 개선"
    rule = analyze_sentence(text, force_cosmetic=False)
    final = asyncio.run(analyze_with_kobert(text, rule))
    assert rule.suspicion_level == SuspicionLevel.CAUTION
    assert "효능과장" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_quantitative_rule_layer_avoids_one_character_metric_labels():
    keywords, patterns = detect_quantitative_claim("사용 4주 후 속 기미 면적 15.34% 개선")
    assert "효능과장" in patterns
    assert keywords
    assert all(not keyword.startswith("사 15.34%") for keyword in keywords)
