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


def test_benign_lasting_makeup_phrase_stays_normal():
    text = "지속력 좋은 쿠션입니다"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_24_hour_lasting_claim_stays_caution():
    text = "이 제품은 24시간 메이크업 지속을 돕니다"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.CAUTION
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_doctor_developed_formula_is_flagged_as_suspicious():
    text = "의사가 개발한 포뮬러"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_barrier_claim_is_treated_as_cosmetic_domain():
    text = "장벽 강화에 도움을 줍니다"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_lifting_test_phrase_is_not_rejected_as_non_domain():
    text = "리프팅 개선 테스트 완료"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_clinical_test_phrase_is_caution():
    text = "임상 테스트 완료 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.CAUTION
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_human_application_study_phrase_is_caution():
    text = "인체 적용 시험 완료 에센스"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.CAUTION
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_hospital_codeveloped_claim_is_suspicious():
    text = "병원 공동개발 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_clinically_proven_claim_is_suspicious():
    text = "임상적으로 주름 개선이 입증된 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "검증오인" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_human_study_verified_claim_is_suspicious():
    text = "인체 적용 시험으로 미백 효과 검증"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "검증오인" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_fda_certified_effect_claim_is_suspicious():
    text = "FDA 인증 미백 효과"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_mfds_certified_wrinkle_claim_is_suspicious():
    text = "식약처 인증 주름 개선"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_fda_registered_ingredient_phrase_stays_normal():
    text = "FDA 등록 원료 사용"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_domestic_number_one_claim_is_caution():
    text = "국내 1위 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "비교우위" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_industry_first_claim_is_caution():
    text = "업계 최초 미백 세럼"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "비교우위" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_compared_to_competitors_twice_claim_is_caution():
    text = "타사 대비 2배 보습"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "비교우위" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_better_than_existing_product_claim_is_caution():
    text = "기존 제품보다 더 우수한 앰플"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "비교우위" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_best_claim_is_caution():
    text = "BEST 미백 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "비교우위" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_number_one_claim_is_caution():
    text = "No.1 보습 앰플"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "비교우위" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_review_certified_claim_is_suspicious():
    text = "후기 인증 완료 세럼"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_before_after_photo_claim_is_suspicious():
    text = "전후 사진 공개 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_user_review_certified_claim_is_suspicious():
    text = "사용자 리뷰 인증 쿠션"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS

def test_allowed_hair_loss_relief_phrase_stays_normal():
    text = "탈모 증상 완화에 도움을 주는 화장품"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_allowed_skin_damage_prevention_phrase_stays_normal():
    text = "피부 손상 예방 및 개선"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL


def test_allowed_acne_prone_skin_phrase_stays_normal():
    text = "여드름성 피부 사용 적합"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL
