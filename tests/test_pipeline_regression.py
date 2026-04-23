import asyncio

import pytest

from analysis.models.schemas import SuspicionLevel
from analysis.services.ad_domain_filter import predict_ad, predict_cosmetic
from analysis.services.ai_analyzer import analyze_with_kobert, _resolve_kobert_level
from analysis.services.rule_engine import analyze_sentence, should_run_kobert


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

ALLOW_CONFLICT_CASES = [
    "주름개선 도움",
    "항균(인체세정용 제품에 한함)",
    "보습을 통해 피부건조에 기인한 가려움의 일시적 완화에 도움",
    "우유 엑소좀, 식물 엑소좀 등",
    "식물 엑소좀, 우유 엑소좀 등",
    "피부 건조에 기인한 가려움 완화",
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


@pytest.mark.parametrize("text", ALLOW_CONFLICT_CASES)
def test_allow_conflict_sentences_stay_normal(text: str):
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL
    assert should_run_kobert(rule) is False


def test_whitelist_allow_sentence_stays_normal():
    text = "피부 보습에 도움을 주는 크림입니다."
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert final.suspicion_level == SuspicionLevel.NORMAL
    assert should_run_kobert(rule) is False


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


def test_context_allow_sentence_is_forwarded_to_kobert():
    text = "피부과 테스트 완료 세럼"
    rule = analyze_sentence(text, force_cosmetic=True)
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert "피부과 테스트 완료" in rule.matched_keywords
    assert should_run_kobert(rule) is True


def test_non_domain_sentence_skips_kobert():
    text = "오늘 배송 시작합니다"
    rule = analyze_sentence(text, force_cosmetic=False)
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert should_run_kobert(rule) is False


def test_non_domain_filter_does_not_hide_forbidden_claim():
    text = "식약처에서 실제로 발모효과를 입증받은 기능성 제품입니다."
    rule = analyze_sentence(text, force_cosmetic=False)
    assert "관련 없는 문구" not in rule.reason
    assert rule.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_non_domain_filter_does_not_hide_mts_claim():
    text = "MTS 기기와 함께 사용하면서 진피층 끝까지 침투"
    rule = analyze_sentence(text, force_cosmetic=False)
    assert "관련 없는 문구" not in rule.reason
    assert rule.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


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


def test_fda_certification_warranty_phrase_is_suspicious():
    text = "미국 FDA 인증·보증"
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


def test_wrinkle_relief_or_improvement_phrase_is_caution():
    text = "주름 완화 또는 개선"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_hair_fall_reduction_phrase_is_caution():
    text = "빠지는 모발을 감소"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_breast_skin_lifting_phrase_is_caution():
    text = "가슴 피부 탄력/리프팅 개선"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level == SuspicionLevel.CAUTION


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


def test_hair_loss_prevention_variant_is_not_missed():
    text = "풍성한 거품 모발강화 & 탈모방지 효과 샴푸!"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_skin_damage_repair_phrase_is_not_treated_as_normal():
    text = "피부 손상 복구"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_doctor_developed_product_phrase_is_flagged():
    text = "피부과 의사가 제품에 대한 자문의원으로 제품 개발"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_hospital_recommendation_phrase_is_flagged():
    text = "OO 병원에서 추천하는 화장품"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_generic_disease_progression_phrase_is_not_dropped_as_non_domain():
    text = "질병으로 진행 안되게 도움을 주는"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_medical_institution_technology_claim_is_not_missed():
    text = "OO 의료기관의 첨단기술로 탄생한 화장품"
    rule, final = asyncio.run(_run_pipeline(text))
    assert rule.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_hospital_only_claim_is_flagged():
    text = "병원용, 병원전용, 피부과전용, 피부과시술용, 약국용, 약국전용 화장품"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "추천보증" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_nonfunctional_cosmetic_claim_with_wrinkle_and_whitening_is_flagged():
    text = "기능성화장품이 아님에도 주름개선, 미백 등에 효과가"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "기능성오인" in rule.matched_patterns
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_hair_growth_metric_claim_is_not_missed():
    text = "모발 생성이 426% 증가"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "고위험기능성오인" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_detox_claim_is_not_treated_as_normal():
    text = "피부 독소 제거(디톡스)"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_botox_claim_is_not_treated_as_normal():
    text = "바르는 Botox 크림"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "고위험기능성오인" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_bacteria_99_percent_suppression_claim_is_flagged():
    text = "즉각 세균 99.9% 억제율"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_deep_dermis_delivery_claim_is_flagged():
    text = "두피 진피층까지 영양 전달"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_hair_growth_factor_metric_claim_is_flagged():
    text = "모발성장 조절인자 VEGF, KGF 426% 증가"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "고위험기능성오인" in rule.matched_patterns
    assert final.suspicion_level == SuspicionLevel.SUSPICIOUS


def test_pimple_improvement_claim_is_not_short_normal():
    text = "뾰루지 개선"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_breast_enhancement_claim_is_not_short_normal():
    text = "가슴 탄력·확대"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_cosmeceutical_claim_is_not_short_normal():
    text = "코스메슈티컬(cosmeceutical)"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_itch_relief_claim_is_not_treated_as_normal():
    text = "피부 가려움 완화"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_spot_and_elasticity_reset_ad_copy_is_caution():
    text = "탄력 저하 기미잡티 아무리 쉬어도 지쳐 보이는 피부 CEQ로 리셋하세요"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level == SuspicionLevel.CAUTION


def test_vaginal_dryness_improvement_claim_is_not_non_domain():
    text = "질 건조증 개선"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_hair_papilla_claim_is_not_non_domain():
    text = "모유두 연구를 거듭하여"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_endocrine_action_claim_is_not_non_domain():
    text = "호르몬 분비촉진 등 내분비 작용"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_trace_removal_claim_is_not_non_domain():
    text = "홍티의 흔적을 없애준다"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_vegan_no_animal_testing_claim_is_not_non_domain():
    text = "동물 실험 없이 만든 비건 인증 제품임"
    rule, final = asyncio.run(_run_pipeline(text))
    assert "관련 없는 문구" not in rule.reason
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_no_animal_testing_x_claim_is_not_normal():
    text = "화학 성분을 넣지 동물 실험 X"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_peel_off_scab_claim_is_not_normal():
    text = "기미 부분의 피부를 녹여 딱지를 생기게 한 뒤 뜯어내는 제품"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_roll_stamp_claim_is_not_normal():
    text = "3D 롤스탬프 사용 후 에센샷 앰플"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_non_skin_body_part_medical_efficacy_claim_is_not_normal():
    text = "관절, 림프선 등 피부 이외 신체 특정 부위에 사용하여 의학적 효능, 효과 표방"
    rule, final = asyncio.run(_run_pipeline(text))
    assert final.suspicion_level in {SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS}


def test_kobert_thresholds_keep_normal_sentence_below_caution_threshold():
    rule = analyze_sentence("일반 문구", force_cosmetic=True)
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert _resolve_kobert_level(rule, 0.45) == SuspicionLevel.NORMAL


def test_kobert_thresholds_raise_normal_sentence_to_suspicious_at_high_score():
    rule = analyze_sentence("일반 문구", force_cosmetic=True)
    assert rule.suspicion_level == SuspicionLevel.NORMAL
    assert _resolve_kobert_level(rule, 0.82) == SuspicionLevel.SUSPICIOUS


def test_kobert_downgrades_rule_suspicious_when_score_is_moderate():
    rule = analyze_sentence("이 크림은 아토피를 치료합니다.", force_cosmetic=True)
    assert rule.suspicion_level == SuspicionLevel.SUSPICIOUS
    assert _resolve_kobert_level(rule, 0.52) == SuspicionLevel.CAUTION
