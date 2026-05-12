from analysis.models.schemas import SuspicionLevel
from analysis.services.rule_engine import analyze_sentence


def test_functional_growth_claims_are_not_dropped_as_non_domain() -> None:
    assert analyze_sentence("(속)눈썹이 자란다").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("호르몬 분비 촉진").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("피하지방 분해").suspicion_level == SuspicionLevel.SUSPICIOUS


def test_high_risk_functional_growth_claims_are_escalated_to_suspicious() -> None:
    assert analyze_sentence("모발 두께 증가").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("세포성장 촉진").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("얼굴 크기 축소").suspicion_level == SuspicionLevel.SUSPICIOUS


def test_spacing_and_particle_variants_are_not_missed() -> None:
    assert analyze_sentence("모발굵기 증가").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("가는 모발 굵기 증가").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("모발의 두께를 증가").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("세포 성장을 촉진").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("체형변화").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("얼굴 크기가 작아진다").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("속눈썹, 눈썹이 자람").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("체내 노폐물 제거").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("유익균의 균형 보호").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("땀 발생을 억제").suspicion_level == SuspicionLevel.SUSPICIOUS


def test_clear_risky_normal_misses_are_no_longer_dropped() -> None:
    assert analyze_sentence("모발 등의 성장을 촉진 또는 억제").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("피부구성 물질(예 : 효소, 콜라겐 등)을 증가, 감소 또는 활성화시킨다").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("주름이 채워지고 속눈썹이 자라는 역주행 대란템").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("모발생장촉진").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("지방볼륨생성").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("통증 경감").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("얼굴 윤곽개선, V라인").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("기능성 화장품 심사·보고하지 않은 제품에서 “식약처 미백 고시 성분 OO 함유” 표현").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("일시적 악화(명현 현상)이 있을 수 있다").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("상처로 인한 반흔을 제거 또는 완화한다").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("피부 나이 n 감소 또는 어려진다는 표현").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("미세먼지 차단, 미세먼지 흡착 방지").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("피부노화 완화, 안티에이징, 피부노화 징후 감소, 피부노화지수 감소").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("여성크림, 성 윤활작용").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("쾌감을 증대").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("질 보습, 질 수축 작용").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("콜라겐 증가, 감소 또는 활성화").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("효소 증가, 감소 또는 활성화").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("기능성 화장품 심사(보고)하지 아니한 제품에 미백, 화이트닝(whitening), 주름(링클, wrinkle) 개선, 자외선(UV)차단 등 기능성 관련 표현").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("기능성화장품으로 심사(보고)하지 아니한 제품에 ‘식약처 미백 고시 성분 OO 함유’ 등의 표현").suspicion_level == SuspicionLevel.SUSPICIOUS


def test_detox_and_regeneration_claims_escalate_only_with_strong_context() -> None:
    assert analyze_sentence("피부 독소 제거(디톡스)").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("피부재생").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("피부세포를 재생시키는 인체 줄기세포의 마법").suspicion_level == SuspicionLevel.SUSPICIOUS


def test_remaining_boundary_functional_claims_follow_latest_policy() -> None:
    assert analyze_sentence("유익균 균형 보호").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("질내 산도 유지").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("땀 억제").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("세포 활력(증가)").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("셀룰라이트 감소").suspicion_level == SuspicionLevel.SUSPICIOUS
    assert analyze_sentence("피부 손상 회복").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("피부 손상 복구").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("피부결 20% 개선").suspicion_level == SuspicionLevel.CAUTION
    assert analyze_sentence("피부 면역력 향상").suspicion_level == SuspicionLevel.CAUTION


def test_allowlisted_regulatory_functional_product_phrase_stays_normal() -> None:
    assert analyze_sentence("주름 개선 기능성 화장품으로 허가받은 제품입니다.").suspicion_level == SuspicionLevel.NORMAL


def test_allowlisted_regulatory_functional_phrases_do_not_trigger_endorsement() -> None:
    result = analyze_sentence("식약처 허가 기능성 화장품 (미백에 도움)에 도움을 주는 성분 함유 에센스.")
    assert "추천보증" not in result.matched_patterns


def test_help_level_functional_claims_can_be_allowlisted() -> None:
    result = analyze_sentence("주름 개선에 도움")
    assert result.suspicion_level == SuspicionLevel.NORMAL


def test_help_level_functional_claim_variants_can_be_allowlisted() -> None:
    assert analyze_sentence("식약처 허가 기능성 화장품 (주름 개선에 도움)입니다.").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("아데노신 성분이 주름 개선에 도움을 줍니다.").suspicion_level == SuspicionLevel.NORMAL


def test_allowlisted_functional_phrases_with_duration_do_not_escalate_only_for_soft_caution() -> None:
    assert analyze_sentence("하루 종일 지속되는 피부 보습에 도움 효과.").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("하루 종일 지속되는 식약처 허가 기능성 화장품 (주름 개선에 도움) 효과.").suspicion_level == SuspicionLevel.NORMAL


def test_allowlisted_functional_phrases_keep_normal_with_wrapper_copy() -> None:
    assert analyze_sentence("매일 사용하기 좋은 빠지는 모발 감소에 도움 케어 로션.").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("피부를 촉촉하게 가꾸어주는 빠지는 모발 감소에 도움 제품.").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("가볍고 산뜻하게 식약처 허가 기능성 화장품 (주름 개선에 도움)을 도와주는 크림.").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("집중 케어 여드름성 피부 사용 적합").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("피부과학 기반 여드름성 피부에 사용에 적합").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("여드름성 피부 사용 적합 인체 적용 시험 완료").suspicion_level == SuspicionLevel.NORMAL


def test_regulatory_ingredient_phrases_do_not_trigger_endorsement() -> None:
    assert analyze_sentence("식약처 허가 원료 함유").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("식약처 허가 성분으로 구성된 피부결 보습에 도움 화장품.").suspicion_level == SuspicionLevel.NORMAL


def test_contextual_notice_and_exosome_listing_stay_normal() -> None:
    assert analyze_sentence("피부과학 기반 우유 엑소좀, 식물 엑소좀 등").suspicion_level == SuspicionLevel.NORMAL
    assert analyze_sentence("피부과학 기반 화장품 허위·과대 광고의 근원적 문제를 해결하기 위해 최선을 다하겠다").suspicion_level == SuspicionLevel.NORMAL

