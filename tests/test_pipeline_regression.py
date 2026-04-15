import asyncio

import pytest

from analysis.models.schemas import SuspicionLevel
from analysis.services.ad_domain_filter import predict_ad, predict_cosmetic
from analysis.services.ai_analyzer import analyze_with_kobert
from analysis.services.rule_engine import analyze_sentence


CASES = [
    (
        "이 크림은 아토피를 치료합니다.",
        SuspicionLevel.SUSPICIOUS,
    ),
    (
        "단 7일 만에 기미 완전 제거, 100% 효과 보장.",
        SuspicionLevel.SUSPICIOUS,
    ),
    (
        "피부 보습에 도움을 주는 크림입니다.",
        SuspicionLevel.NORMAL,
    ),
    (
        "피부과 테스트 완료, 촉촉한 사용감의 로션입니다.",
        SuspicionLevel.NORMAL,
    ),
    (
        "의사가 추천하는 세럼으로 주름이 반드시 사라집니다.",
        SuspicionLevel.SUSPICIOUS,
    ),
    (
        "줄기세포 기술로 피부를 재생시키는 앰플입니다.",
        SuspicionLevel.CAUTION,
    ),
    (
        "피부 각질 100프로 제거",
        SuspicionLevel.SUSPICIOUS,
    ),
    (
        "피부 보습 건조 모두 해결",
        SuspicionLevel.CAUTION,
    ),
]


async def _run_pipeline(text: str):
    cosmetic = predict_cosmetic(text)
    rule = analyze_sentence(text, force_cosmetic=False if cosmetic is None else cosmetic[0])
    final = await analyze_with_kobert(text, rule)
    return rule, final


@pytest.mark.parametrize(("text", "expected_level"), CASES)
def test_pipeline_regression_levels(text: str, expected_level: SuspicionLevel):
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level == expected_level, (
        f"text={text!r}, rule={rule.suspicion_level.value}, final={final.suspicion_level.value}, "
        f"reason={final.reason}"
    )


def test_whitelist_allow_sentence_stays_normal():
    text = "피부 보습에 도움을 주는 크림입니다."
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_doctor_recommendation_is_not_whitelisted():
    text = "의사가 추천하는 세럼으로 주름이 반드시 사라집니다."
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_ad_filter_identifies_obvious_ad_sentence():
    text = "단 7일 만에 기미 완전 제거, 100% 효과 보장."
    ad_pred = predict_ad(text)
    assert ad_pred is not None
    assert ad_pred[0] is True


def test_stem_cell_regeneration_keeps_minimum_caution():
    text = "줄기세포 기술로 피부를 재생시키는 앰플입니다."
    rule, final = asyncio.run(_run_pipeline(text))
    assert "첨단기술오인" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION
    assert final.score >= 0.35
