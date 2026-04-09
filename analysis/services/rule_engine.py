import re
from analysis.models.schemas import SuspicionLevel, SentenceResult

# ────────────────────────────────────────────
# 화이트리스트 (허용 표현)
# 출처: 정제 시트 기준 라벨=1 (허용문구)
# 이 표현이 포함된 문장은 규칙기반 의심 탐지에서 제외
# ────────────────────────────────────────────

WHITELIST_KEYWORDS: list[str] = [
    # 식약처 허가 기능성 표현
    "식약처 허가 기능성 화장품",
    "기능성 화장품 심사",
    "기능성 화장품 (미백에 도움)",
    "기능성 화장품 (주름 개선에 도움)",
    "기능성 화장품 (자외선 차단)",
    # 피부과 테스트
    "피부과 테스트 완료",
    "피부과 임상 테스트 완료",
    "피부과 테스트 마친",
    # 허용 효능 표현 (도움을 주는 수준)
    "피부 보습에 도움",
    "피부 진정에 도움",
    "피부 결 정돈에 도움",
    "수분 공급에 도움",
    "피부에 생기를 부여",
    "피부를 건강하게 가꾸는 데 도움",
    "촉촉한 사용감",
    "산뜻한 마무리감",
    "모발 손상 개선에 도움",
    "빠지는 모발 감소에 도움",
    "탈모 증상 완화에 도움",
    # 자외선 차단 (허용)
    "자외선 차단 기능",
    "SPF50+/PA++++",
    "SPF",
    "PA+",
    # 천연·유기농 지수 (ISO 기반 허용 표현)
    "ISO 16128",
    "천연(유래)지수",
    "천연유래지수",
    "ISO 천연",
    "유기농 지수",
    # 인증 관련 허용 표현
    "COSMOS 인증",
    "시험검사기관",
    "ISO 인증",
    # 기타 허용 표현
    "가려준다",
    "탄력/리프팅 개선",
    "건조한 피부를 촉촉하게",
    "건조함 등으로 인한 트러블",
    "피부장벽 강화에 도움",
    "피부 수분 유지에 도움",
    # 일반 제품명/카테고리 표현 (KoBERT 과탐지 방지)
    "보습로션", "아기로션", "아기크림", "아기보습",
    "로션", "크림", "세럼", "앰플", "에센스", "토너",
    "선크림", "선스크린", "클렌저", "폼클렌징",
    "공유하기", "URL복사", "신고하기", "본문 바로가기",
    "카테고리 이동", "이웃추가", "MY메뉴",
]

# ────────────────────────────────────────────
# 금지 키워드 사전
# 출처: 식품의약품안전처 화장품 표시·광고 가이드라인,
#       공정거래위원회 표시광고법 위반 사례집
# ────────────────────────────────────────────

FORBIDDEN_KEYWORDS: dict[str, list[str]] = {

    # ── 의약품 오인 / 질병 치료·예방 표현 ──────────────────
    "의약품오인": [
        # 치료·완치 관련
        "치료", "완치", "치유", "치료제", "치료 효과", "치료에 도움",
        "의약품", "약효", "약리", "처방",
        "예방", "증상 개선", "증상 완화",
        # 질환명 직접 연결
        "아토피", "아토피 치료", "아토피 개선", "아토피 완화",
        "피부염", "피부염 치료", "피부염 개선", "피부염 완화",
        "알레르기", "알레르기 치료", "알레르기 억제", "알레르기 완화",
        "습진", "건선", "여드름", "질염",
        "상처 치유", "상처 회복", "흉터 제거", "흉터 치료",
        # 항균·소독 계열
        "항균", "살균", "멸균", "소독", "항바이러스",
        "세균 제거", "바이러스 억제",
        # 염증 관련
        "염증 억제", "염증 완화", "염증 개선", "염증 치료",
        "소염", "항염",
        # 의사·병원 관련
        "의사 추천", "의사 인증", "의사 처방", "의사 권장",
        "병원 추천", "병원 공인", "병원 처방",
        "피부과 처방", "피부과 인증",
        "전문의 추천", "전문의 인증", "전문의 처방",
        "임상 효과", "임상 입증", "임상 확인", "임상 검증",
        "의학적 효과", "의학적 검증",
    ],

    # ── 효능 과장 / 절대적·단정적 표현 ────────────────────
    "효능과장": [
        # 수치 단정
        "100%", "100% 효과", "100% 안전", "100% 천연",
        "200%", "300%",
        # 즉각성 과장
        "즉시 효과", "즉각 효과", "즉각 개선", "즉시 개선",
        "즉시 완화", "바로 효과", "단번에",
        # 기간 단정
        "7일 만에", "3일 만에", "하루 만에", "일주일 만에",
        "2주 만에", "한 달 만에",
        "며칠 만에", "만에 끝",
        # 보장·확신 표현
        "효과 보장", "결과 보장", "보장합니다", "보장된",
        "반드시 효과", "반드시 개선", "확실히 효과",
        "확실한 효과", "확실히 좋아진", "보장",
        # 완벽·전체 표현
        "완벽한", "완전히", "완전 제거", "완전 개선",
        "전부 없애", "전부 제거",
        # 최상급 과장
        "최강", "세계 최고", "세계 최초", "국내 유일",
        "업계 최초", "국내 최초", "아시아 최초",
        "압도적", "독보적",
        # 부정적 단정
        "무조건", "절대적", "절대 효과",
    ],

    # ── 기능성 화장품 오인 (심사 없이 기능성 주장) ─────────
    "기능성오인": [
        # 주름 관련
        "주름 제거", "주름 없애", "주름 완전 제거",
        "주름 치료", "주름 없어짐",
        # 미백·색소 관련
        "기미 제거", "기미 완전 제거", "잡티 제거",
        "색소 제거", "멜라닌 제거", "미백 치료",
        "피부톤 완전 개선",
        # 모공·탄력 관련
        "모공 축소", "모공 없앰", "모공 제거",
        # 탈모·발모 관련
        "탈모 방지", "탈모 예방", "탈모 치료",
        "발모 촉진", "모발 재생", "두피 재생",
        # 체중·체형 관련
        "체중 감소", "다이어트 효과", "지방 제거",
        "셀룰라이트 제거", "체지방 감소",
        # 세포·재생 관련
        "세포 재생", "피부 재생", "피부세포 재생",
        "피부 복원", "콜라겐 생성 촉진", "콜라겐 재생",
        "줄기세포 재생", "DNA 복구",
    ],

    # ── 안전성 단정 표현 ────────────────────────────────
    "안전성단정": [
        "부작용 없음", "부작용 없는", "부작용 전혀 없", "부작용이 전혀", "부작용 없이", "부작용이 없", "부작용이 전혀",
        "부작용 없이", "부작용이 없",
        "완전 무해", "완전 안전", "완전히 안전",
        "무자극", "무자극 인증", "자극 없음", "자극 전혀 없", "자극이 없", "자극이 없",
        "알레르기 없음", "알레르기 유발 없음",
        "영구적", "영구 지속", "평생 지속",
        "모든 피부 타입", "모든 피부에 사용 가능",
        "누구나 사용 가능", "어떤 피부도",
    ],

    # ── 추천·보증·인증 표현 ─────────────────────────────
    "추천보증": [
        # 전문가 추천
        "의사 추천", "피부과 추천", "전문의 추천",
        "약사 추천", "한의사 추천",
        # 인증·공인
        "의사 인증", "병원 공인", "특허 효과",
        "식약처 인증", "식약처 허가", "식약청 인증",
        "FDA 인증", "유럽 인증",
        # 수상·선정
        "수상 제품", "대상 수상", "올해의 제품",
        "베스트셀러 인증", "소비자 선정",
        # 후기·체험 유도
        "후기 인증", "사용자 인증", "실사용 인증",
        "효과 인증", "전후 사진 인증",
        "사용해보니", "써보니", "직접 사용", "사라졌네요", "체험담",
    ],

    # ── 비교 우위 단정 표현 ─────────────────────────────
    "비교우위": [
        "타사 대비", "경쟁사보다", "시중 제품보다",
        "기존 제품보다 월등", "압도적으로 뛰어난",
        "가장 효과적인", "가장 빠른", "가장 강력한",
    ],

    # ── 첨단 기술·재생 과대 주장 (기능성 오인 세부) ─────────
    "첨단기술오인": [
        "DNA", "줄기세포", "엑소좀", "세포 재생", "세포 활성",
        "피부세포", "재생 기술", "차세대 기술", "나노 입자",
        "진피층 침투", "피부 깊숙한 층", "MTS", "미세침", "필러급",
    ],
}

# ────────────────────────────────────────────
# 주의 키워드 (과장 가능성 있으나 단정 어려운 표현)
# ────────────────────────────────────────────

# ────────────────────────────────────────────
# 화장품 도메인 명사 (관련성 필터)
# 이 중 하나라도 포함돼야 분석 대상으로 처리
# 해당 명사가 없는 문장은 화장품 광고와 무관한 것으로 보고 정상 처리
# ────────────────────────────────────────────

COSMETIC_DOMAIN_NOUNS: frozenset[str] = frozenset({
    # 제품 종류
    "크림", "로션", "세럼", "앰플", "에센스", "토너", "스킨", "미스트",
    "선크림", "선스크린", "선블록", "클렌저", "클렌징", "폼클렌징",
    "마스크팩", "시트마스크", "팩", "아이크림", "립밤", "립크림",
    "파운데이션", "비비크림", "쿠션", "컨실러", "파우더",
    "샴푸", "컨디셔너", "트리트먼트", "헤어에센스", "헤어오일",
    "바디로션", "바디크림", "바디워시",
    # 화장품·뷰티 범주어
    "화장품", "뷰티", "스킨케어", "기초화장품", "색조화장품",
    "메이크업", "코스메틱", "화장",
    # 피부·두피·모발
    "피부", "두피", "모발", "모공", "주름", "미백", "보습",
    "탄력", "피부톤", "피부결", "피부장벽", "각질", "탈모",
    # 성분·제형
    "성분", "제형", "함유", "원료", "추출물",
    # 기능성
    "자외선차단", "기능성화장품",
})


def is_cosmetic_related(sentence: str) -> bool:
    """화장품·뷰티 도메인과 관련된 문장인지 확인"""
    return any(noun in sentence for noun in COSMETIC_DOMAIN_NOUNS)


CAUTION_KEYWORDS: list[str] = [
    # 재생·회복 계열
    "재생", "피부 재생", "손상 회복", "피부 회복",
    "리페어", "리커버리",
    # 줄기세포·첨단 성분
    "줄기세포", "EGF", "성장인자", "펩타이드 재생",
    # 리프팅·탄력
    "리프팅", "탄력 강화", "리프팅 효과", "V라인",
    "피부 당김", "피부 조임",
    # 빠른 효과 암시
    "즉시", "빠른 효과", "빠르게 개선", "놀라운 변화",
    "단기간", "빠른 시간 내",
    # 혁신·과학 과장
    "혁신적", "획기적", "신개념", "차세대", "차세대 기술",
    "과학적으로 증명", "과학적 입증",
    # 천연·유기농 과장
    "100% 천연", "완전 천연", "순수 천연",
    "100% 유기농", "완전 무독성",
    # 기타 과장 가능 표현
    "피부 나이 역전", "피부 나이를 되돌리",
    "동안 피부 보장", "동안 효과",
    "엑소좀", "피부나이", "나노 입자",
]


# ────────────────────────────────────────────
# 패턴 태그 → 한국어 설명 매핑
# ────────────────────────────────────────────

_PATTERN_DESC: dict[str, str] = {
    "의약품오인":  "의약품으로 오인될 수 있는 표현",
    "효능과장":   "효능을 과장하거나 절대적으로 단정하는 표현",
    "기능성오인":  "기능성 화장품 심사 없이 기능성을 주장하는 표현",
    "안전성단정":  "안전성을 근거 없이 단정하는 표현",
    "추천보증":   "전문가 추천·인증을 주장하는 표현",
    "비교우위":   "근거 없이 타사 대비 우위를 주장하는 표현",
    "첨단기술오인": "첨단 기술·세포 재생을 근거 없이 강조하는 표현",
}


# ────────────────────────────────────────────
# 분석 함수
# ────────────────────────────────────────────

def analyze_sentence(sentence: str) -> SentenceResult:
    """1차 규칙기반 엔진: 문장 하나를 분석하여 SentenceResult 반환"""

    # ── 화이트리스트 체크 ───────────────────────────────────
    for wl_kw in WHITELIST_KEYWORDS:
        if wl_kw in sentence:
            return SentenceResult(
                sentence=sentence,
                suspicion_level=SuspicionLevel.NORMAL,
                matched_keywords=[],
                matched_patterns=[],
                reason="허용된 표현이 포함되어 있습니다.",
                score=0.0,
            )

    # ── 도메인 관련성 필터 ─────────────────────────────────
    if not is_cosmetic_related(sentence):
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason="화장품·뷰티 광고와 관련 없는 문구입니다.",
            score=0.0,
        )

    matched_keywords: list[str] = []
    matched_patterns: list[str] = []

    # 금지 키워드 검사
    for pattern_tag, keywords in FORBIDDEN_KEYWORDS.items():
        for kw in keywords:
            if kw in sentence:
                matched_keywords.append(kw)
                if pattern_tag not in matched_patterns:
                    matched_patterns.append(pattern_tag)

    # 주의 키워드 검사
    caution_matches = [kw for kw in CAUTION_KEYWORDS if kw in sentence]
    matched_keywords.extend(caution_matches)

    # 의심도 결정 + 규칙 기반 연속 점수
    if matched_patterns:
        level = SuspicionLevel.SUSPICIOUS
        # 패턴 수에 따라 점수 차등 (0.70 ~ 0.90)
        base_score = round(min(0.70 + len(matched_patterns) * 0.05, 0.90), 3)
        reason = _build_reason(matched_patterns, matched_keywords)
    elif caution_matches:
        level = SuspicionLevel.CAUTION
        # 주의 키워드 수에 따라 점수 차등 (0.35 ~ 0.55)
        base_score = round(min(0.35 + len(caution_matches) * 0.05, 0.55), 3)
        kw_display = ", ".join(f"'{kw}'" for kw in caution_matches[:3])
        reason = f"{kw_display} 표현은 효과를 과장하거나 오인을 유발할 가능성이 있어 주의가 필요합니다."
    else:
        level = SuspicionLevel.NORMAL
        base_score = 0.0
        reason = "특별히 의심되는 표현이 발견되지 않았습니다."

    return SentenceResult(
        sentence=sentence,
        suspicion_level=level,
        matched_keywords=list(set(matched_keywords)),
        matched_patterns=matched_patterns,
        reason=reason,
        score=base_score,
    )


_PATTERN_REASON_TEMPLATE: dict[str, str] = {
    "의약품오인":  "'{kw}' 표현은 화장품에서 사용이 금지된 의약품 오인 표현입니다. 화장품은 질병의 치료·예방을 표방할 수 없습니다.",
    "효능과장":   "'{kw}' 표현은 효과를 단정하거나 과장하는 표현으로, 소비자를 오인하게 할 수 있습니다.",
    "기능성오인":  "'{kw}' 표현은 식약처 기능성 심사 없이 기능성을 주장하는 표현으로, 허가받지 않은 효능 광고에 해당할 수 있습니다.",
    "안전성단정":  "'{kw}' 표현은 과학적 근거 없이 안전성을 단정하는 표현입니다. 화장품은 부작용이 없다고 단정할 수 없습니다.",
    "추천보증":   "'{kw}' 표현은 전문가 추천·인증을 주장하는 표현으로, 실제 근거 없이 사용 시 허위광고에 해당할 수 있습니다.",
    "비교우위":   "'{kw}' 표현은 근거 없이 타사 대비 우위를 주장하는 표현으로, 비교광고 기준에 위반될 수 있습니다.",
}

def _build_reason(patterns: list[str], keywords: list[str]) -> str:
    """패턴 태그 + 감지 키워드를 조합해 문구별 맞춤 이유 생성"""
    if not patterns:
        return "특별히 의심되는 표현이 발견되지 않았습니다."

    reasons = []
    used_keywords = set()

    for pattern in patterns:
        template = _PATTERN_REASON_TEMPLATE.get(pattern)
        if not template:
            continue

        # 해당 패턴에 속하는 키워드 중 아직 사용 안 된 것 선택
        pattern_keywords = FORBIDDEN_KEYWORDS.get(pattern, [])
        matched = [kw for kw in keywords if kw in pattern_keywords and kw not in used_keywords]

        if matched:
            kw_display = ", ".join(f"'{kw}'" for kw in matched[:2])  # 최대 2개
            reason = template.replace("'{kw}'", kw_display)
            used_keywords.update(matched[:2])
        else:
            # 해당 패턴 키워드가 없으면 감지된 키워드 중 첫 번째 사용
            remaining = [kw for kw in keywords if kw not in used_keywords]
            if remaining:
                kw_display = f"'{remaining[0]}'"
                reason = template.replace("'{kw}'", kw_display)
                used_keywords.add(remaining[0])
            else:
                reason = template.replace("'{kw}' 표현은", "해당 표현은")

        reasons.append(reason)

    return " / ".join(reasons)


def calculate_overall_score(results: list[SentenceResult]) -> tuple[float, SuspicionLevel]:
    """전체 의심도 점수 및 레벨 계산 (연속 점수 기반 가중 평균)"""
    if not results:
        return 0.0, SuspicionLevel.NORMAL

    scores = sorted([r.score for r in results], reverse=True)
    n = len(scores)

    # 상위 1/3 문장에 가중치 2배 → 의심 문장이 전체 점수에 확실히 반영
    k = max(1, n // 3)
    weighted_sum = sum(scores[:k]) * 2.0 + sum(scores[k:])
    total_weight = k * 2.0 + (n - k)
    overall = weighted_sum / total_weight

    # 레벨 결정
    if overall >= 0.55:
        level = SuspicionLevel.SUSPICIOUS
    elif overall >= 0.20:
        level = SuspicionLevel.CAUTION
    else:
        level = SuspicionLevel.NORMAL

    # 고신뢰 의심 문장(score >= 0.80, 패턴 2개 이상)이 있으면 최소 주의 보장
    if any(r.suspicion_level == SuspicionLevel.SUSPICIOUS and r.score >= 0.80 for r in results):
        if level == SuspicionLevel.NORMAL:
            level = SuspicionLevel.CAUTION

    return round(overall, 3), level
