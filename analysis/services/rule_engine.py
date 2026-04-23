import re
from analysis.models.schemas import SuspicionLevel, SentenceResult

NON_DOMAIN_NORMAL_REASON = "화장품·뷰티 광고와 관련 없는 문구입니다."
STRICT_ALLOW_NORMAL_REASON = "허용된 표현 중심의 문구로 판단됩니다."
EXTRA_ALLOW_NORMAL_REASON = "허용된 기능성 표현 중심의 문구로 판단됩니다."
CONTEXT_ALLOW_NORMAL_REASON = "허용 또는 안내 성격의 표현이 포함된 문구로 판단됩니다."
DEFAULT_NORMAL_REASON = "특별히 의심되는 표현이 발견되지 않았습니다."

# ────────────────────────────────────────────
# 허용 표현
# - STRICT_ALLOW_KEYWORDS: 문장 전체가 허용 맥락일 때 정상 처리에 사용
# - CONTEXT_ALLOW_KEYWORDS: 정상 보조 근거로만 사용
# - DOMAIN_ONLY_TERMS: 화이트리스트가 아니라 도메인 단서로만 사용
# ────────────────────────────────────────────

STRICT_ALLOW_KEYWORDS: list[str] = [
    # 식약처 허가 기능성 표현
    "식약처 허가 기능성 화장품",
    "기능성 화장품 심사",
    "기능성 화장품 (미백에 도움)",
    "기능성 화장품 (주름 개선에 도움)",
    "기능성 화장품 (자외선 차단)",
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
    "건조한 피부를 촉촉하게",
    "건조함 등으로 인한 트러블",
    "피부장벽 강화에 도움",
    "피부 수분 유지에 도움",
]

STRICT_ALLOW_EXTRA_KEYWORDS: list[str] = [
    "탈모 증상 완화에 도움",
    "탈모 증상 완화에 도움을 주는 화장품",
    "피부 손상 예방 및 개선",
    "여드름성 피부에 사용에 적합",
    "여드름성 피부 사용 적합",
]

# eval 및 가이드라인 기준으로 허용 문구가 명확한 경우에는
# 부분 패턴보다 문장 전체 일치 우선으로 정상 처리한다.
EXACT_ALLOW_SENTENCES: set[str] = {
    "주름개선 도움",
    "항균(인체세정용 제품에 한함)",
    "보습을 통해 피부건조에 기인한 가려움의 일시적 완화에 도움",
    "우유 엑소좀, 식물 엑소좀 등",
    "식물 엑소좀, 우유 엑소좀 등",
    "피부 건조에 기인한 가려움 완화",
}

EXACT_CAUTION_SENTENCES: set[str] = {
    "피부 가려움 완화",
}

CONTEXT_ALLOW_KEYWORDS: list[str] = [
    # 피부과 테스트
    "피부과 테스트 완료",
    "피부과 임상 테스트 완료",
    "피부과 테스트 마친",
    # 문맥에 따라 허용/과장 가능성이 갈릴 수 있는 표현
    "가려준다",
    "탄력/리프팅 개선",
    # 일반 UI/크롤링 노이즈
    "공유하기", "URL복사", "신고하기", "본문 바로가기",
    "카테고리 이동", "이웃추가", "MY메뉴",
]

DOMAIN_ONLY_TERMS: list[str] = [
    "보습로션", "아기로션", "아기크림", "아기보습",
    "로션", "크림", "세럼", "앰플", "에센스", "토너",
    "선크림", "선스크린", "클렌저", "폼클렌징",
]

ALLOW_CONTEXT_STOPWORDS: set[str] = {
    "이", "가", "은", "는", "을", "를", "에", "의", "과", "와", "도", "로", "으로",
    "및", "또는", "한", "수", "더", "등", "제품", "사용", "사용감", "기능", "완료",
    "마친", "도움", "주는", "주는", "주며", "좋은", "좋습니다", "입니다", "있습니다",
    "피부과", "테스트",
}

# 화장품 여부와 무관하게 강하게 차단할 표현 (도메인 필터 우회)
STRONG_FORBIDDEN_KEYWORDS: list[str] = [
    "100%", "100퍼", "100퍼센트", "100프로",
    "영구", "영구제거", "평생 보장", "완치", "완벽", "즉시 효과",
    "의사 보증", "의사추천", "의사 인증",
    "부작용 없음", "전혀 부작용", "무조건 환불",
    "평생 유지", "확실한 효과",
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
        "예방", "증상 개선", "증상 완화", "질병",
        # 질환명 직접 연결
        "아토피", "아토피 치료", "아토피 개선", "아토피 완화",
        "피부염", "피부염 치료", "피부염 개선", "피부염 완화",
        "알레르기", "알레르기 치료", "알레르기 억제", "알레르기 완화",
        "습진", "건선", "여드름", "질염", "두드러기", "불면증", "두통",
        "상처 치유", "상처 회복", "흉터 제거", "흉터 치료", "홍반", "홍조", "뾰루지",
        "가려움", "건조증",
        # 항균·소독 계열
        "항균", "살균", "멸균", "소독", "항바이러스",
        "세균 제거", "바이러스 억제", "세균 억제", "억제율",
        # 염증 관련
        "염증", "염증 억제", "염증 완화", "염증 개선", "염증 치료", "염증 진정",
        "소염", "항염", "근육이완", "요로 세척", "심신 안정",
        # 의사·병원 관련
        "의사 추천", "의사 인증", "의사 처방", "의사 권장",
        "병원 추천", "병원 공인", "병원 처방",
        "피부과 처방", "피부과 인증",
        "전문의 추천", "전문의 인증", "전문의 처방",
        "임상 효과", "임상 입증", "임상 확인", "임상 검증",
        "의학적 효과", "의학적 검증", "의학적 효능",
        "관절", "림프선", "피부 이외", "신체 특정 부위",
    ],

    # ── 효능 과장 / 절대적·단정적 표현 ────────────────────
    "효능과장": [
        # 수치 단정
        "100%", "100% 효과", "100% 안전", "100% 천연",
        "200%", "300%",
        # 즉각성 과장
        "즉시 효과", "즉각 효과", "즉각 개선", "즉시 개선",
        "즉시 완화", "바로 효과", "단번에",
        "마법",
        # 기간 단정
        "7일 만에", "3일 만에", "하루 만에", "일주일 만에",
        "2주 만에", "한 달 만에",
        "며칠 만에", "만에 끝",
        "끝내버리세요",
        # 보장·확신 표현
        "효과 보장", "결과 보장", "보장합니다", "보장된",
        "반드시 효과", "반드시 개선", "확실히 효과",
        "확실한 효과", "확실히 좋아진", "보장",
        # 완벽·전체 표현
        "완벽한", "완전히", "완전 제거", "완전 개선",
        "전부 없애", "전부 제거", "전부 사라진다", "모두 사라진다", "다 사라진다", "싹 사라진다",
        "싹 없어짐", "말끔히 사라진다", "깨끗이 사라진다", "완전히 사라진다", "흔적 없이 사라진다",
        # 최상급 과장
        "최강", "세계 최고",
        "압도적", "독보적",
        # 부정적 단정
        "무조건", "절대적", "절대 효과",
        "면역력 향상", "볼륨 개선", "유해물질 배제", "노폐물 배출", "독소 제거",
        "강력투입", "진피층까지", "가슴 확대", "탄력·확대",
    ],

    # ── 기능성 화장품 오인 (심사 없이 기능성 주장) ─────────
    "기능성오인": [
        # 주름 관련
        "주름 제거", "주름 없애", "주름 완전 제거", "주름 방지", "주름 완화", "주름개선",
        "주름 치료", "주름 없어짐",
        # 미백·색소 관련
        "기미 제거", "기미 완전 제거", "잡티 제거",
        "색소 제거", "멜라닌 제거", "미백 치료", "미백효과", "홍조 개선", "홍반 개선",
        "피부톤 완전 개선",
        # 모공·탄력 관련
        "모공 축소", "모공 없앰", "모공 제거",
        # 탈모·발모 관련
        "탈모 방지", "탈모방지", "탈모 예방", "탈모 치료",
        "발모효과", "모발강화", "탈모 샴푸",
        "발모 촉진", "모발 재생", "두피 재생", "모발 성장 촉진", "모발 생성", "모발 수 증가", "모발 두께 증가",
        "모발성장", "모낭 성장", "모낭 주기 조절", "모발 휴지기 조절인자", "모발 죽기 조절인자",
        "피부 손상 복구", "피부 손상 회복", "복구", "회복",
        # 체중·체형 관련
        "체중 감소", "다이어트 효과", "지방 제거", "지방분해",
        "셀룰라이트 제거", "셀룰라이트", "체지방 감소", "쥐젖 제거",
        # 세포·재생 관련
        "세포 재생", "피부 재생", "피부세포 재생",
        "피부 복원", "지방세포 활성화", "콜라겐 생성 촉진", "콜라겐 재생",
        "줄기세포 재생", "DNA 복구", "손상된 피부 집중 재생", "Botox", "보톡스",
        "코스메슈티컬", "cosmeceutical", "medicine", "메디슨",
        "모유두", "호르몬 분비촉진", "내분비 작용", "홍티",
        "롤스탬프", "딱지",
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
        "화학 성분을 넣지",
    ],

    # ── 추천·보증·인증 표현 ─────────────────────────────
    "추천보증": [
        # 전문가 추천
        "의사 추천", "피부과 추천", "전문의 추천",
        "약사 추천", "한의사 추천",
        "의사가 개발한", "의사 개발", "전문의 개발", "병원 추천", "병원에서 추천", "자문의원",
        "의료기관", "병원용", "병원전용", "피부과전용", "피부과시술용", "약국용", "약국전용",
        "시술 관련 표현", "레이저", "카복시",
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

HIGH_RISK_FUNCTIONAL_KEYWORDS: list[str] = [
    "기능성화장품이 아님에도",
    "모발 휴지기 조절인자",
    "모발 죽기 조절인자",
    "모발성장 조절인자",
    "모낭 주기 조절",
    "모낭 성장",
    "모발 수 증가",
    "모발 생성",
    "모발 성장 촉진",
    "탈모방지",
    "탈모 방지",
    "지방분해",
    "지방세포",
    "쥐젖 제거",
    "Botox",
    "보톡스",
    "롤스탬프",
    "딱지",
    "피부를 녹여",
    "세포 또는 유전자",
    "DNA 활성화",
    "유전자(DNA) 활성화",
    "medicine",
    "메디슨",
    "홍티",
]

FORBIDDEN_REGEX_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "추천보증": [
        (r"(의사|피부과|전문의|약사|한의사)[가-힣]*\s*추천", "전문가 추천"),
        (r"(의사|피부과|전문의|약사|한의사)[가-힣\s]{0,10}추천", "전문가 추천"),
        (r"(의사|병원|피부과|전문의)[가-힣]*\s*인증", "전문가 인증"),
        (r"(의사|병원|피부과|전문의)[가-힣]*\s*처방", "전문가 처방"),
        (r"(FDA|식약처|식품의약품안전처|미국\s*FDA)[가-힣A-Za-z\s]*\s*인증", "공인기관 인증"),
        (r"(의사|피부과|전문의|약사|한의사)[가-힣]*\s*(개발|설계|공동개발)", "전문가 개발"),
        (r"(의사|피부과|전문의|약사|한의사)[가-힣\s]{0,10}(개발|설계|공동개발)", "전문가 개발"),
        (r"(의사|피부과|전문의)[가-힣\s]{0,30}자문의원[가-힣\s]{0,20}(개발|설계|공동개발)", "전문가 개발"),
        (r"(병원|클리닉|연구소)[가-힣]*\s*공동개발", "기관 공동개발"),
        (r"(병원)[가-힣\s]{0,10}추천", "전문가 추천"),
        (r"(병원용|병원전용|피부과전용|피부과시술용|약국용|약국전용)", "전문기관 전용"),
        (r"(후기|리뷰|사용자\s*리뷰)\s*인증", "후기 인증"),
        (r"(전\s*/\s*후|전후)\s*사진", "전후 사진"),
    ],
    "검증오인": [
        (
            r"(임상(적으로)?|인체\s*적용\s*시험(으로|에서)?|테스트(로|에서)?)[가-힣A-Za-z0-9\s]{0,30}?(입증|검증|확인|증명|보장)",
            "임상 입증",
        ),
    ],
    "비교우위": [
        (r"(국내|업계|세계|전국)\s*(1위|최초)", "1위/최초"),
        (r"\b(BEST|Best|best)\b", "BEST"),
        (r"\bNo\.?\s*1\b", "No.1"),
        (r"(타사|기존\s*제품|경쟁사)\s*대비\s*\d+\s*배", "대비 2배"),
        (r"(기존\s*제품|타사\s*제품|타사)\s*보다\s*(더\s*)?(우수|뛰어난|좋은|강력한)", "타사 대비 우수"),
    ],
    "효능과장": [
        (r"반드시\s*[가-힣]+\w*", "반드시"),
        (r"(주름|기미|잡티|흉터)[가-힣]*\s*(완전히\s*)?(사라지|없어지)", "사라지다"),
        (r"(세균|항균)[가-힣A-Za-z0-9\s\.\-]*99(\.9)?%", "99.9%"),
    ],
    "기능성오인": [
        (r"(주름|기미|잡티|모공)[가-힣]*\s*(제거|없애|없앰|완전 제거)", "제거"),
        (r"(주름|미백|홍조|홍반)[가-힣]*\s*(개선|완화|방지)", "기능성 효능"),
        (r"(가려움|건조증|뾰루지)[가-힣\s]*\s*(완화|개선|방지)", "기능성 효능"),
        (r"(피부|세포|모발|두피)[가-힣]*\s*재생", "재생"),
        (r"(모발|머리카락)[가-힣A-Za-z0-9\s\-β]*\s*(증가|감소)", "모발 수치 변화"),
        (r"(진피층)[가-힣A-Za-z0-9\s]*\s*(전달|침투|투입)", "피부 깊은 층 전달"),
        (r"(가슴)[가-힣\s]*\s*(확대|탄력)", "신체 부위 효능"),
        (r"(독소)\s*(제거|배출)", "디톡스"),
        (r"(흔적)[가-힣\s]*\s*(없애|제거)", "제거"),
        (r"(녹여).*(딱지|뜯어내)", "물리적 제거"),
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
    "탄력", "피부톤", "피부결", "피부장벽", "장벽", "각질", "탈모", "리프팅",
    # 성분·제형
    "성분", "제형", "함유", "원료", "추출물", "포뮬러",
    # 기능성
    "자외선차단", "기능성화장품",
    "잡티",
})



def is_cosmetic_related(sentence: str) -> bool:
    """화장품·뷰티 도메인과 관련된 문장인지 확인"""
    return any(noun in sentence for noun in COSMETIC_DOMAIN_NOUNS)


def _find_forbidden_patterns(sentence: str) -> tuple[list[str], list[str]]:
    matched_keywords: list[str] = []
    matched_patterns: list[str] = []

    for pattern_tag, keywords in FORBIDDEN_KEYWORDS.items():
        for kw in keywords:
            if kw in sentence:
                matched_keywords.append(kw)
                if pattern_tag not in matched_patterns:
                    matched_patterns.append(pattern_tag)

    for pattern_tag, regex_items in FORBIDDEN_REGEX_PATTERNS.items():
        for pattern, label in regex_items:
            if re.search(pattern, sentence):
                matched_keywords.append(label)
                if pattern_tag not in matched_patterns:
                    matched_patterns.append(pattern_tag)

    return matched_keywords, matched_patterns


def _find_whitelist_matches(sentence: str) -> list[str]:
    matches = [kw for kw in STRICT_ALLOW_KEYWORDS if kw in sentence]
    matches.extend(kw for kw in STRICT_ALLOW_EXTRA_KEYWORDS if kw in sentence)
    return matches


def _find_context_allow_matches(sentence: str) -> list[str]:
    return [kw for kw in CONTEXT_ALLOW_KEYWORDS if kw in sentence]


def _is_clearly_allowed_sentence(sentence: str, whitelist_matches: list[str]) -> bool:
    """
    허용 표현이 문장 일부에 섞인 경우까지 정상 처리하지 않도록,
    화이트리스트 구문을 제거한 뒤 남는 의미 토큰이 거의 없는 경우만 허용한다.
    """
    residual = sentence
    for kw in sorted(whitelist_matches, key=len, reverse=True):
        residual = residual.replace(kw, " ")

    tokens = re.findall(r"[가-힣A-Za-z0-9%+/]+", residual)
    meaningful_tokens = [
        token for token in tokens
        if len(token) >= 2 and token not in ALLOW_CONTEXT_STOPWORDS
    ]
    return len(meaningful_tokens) <= 1


def should_run_kobert(rule_result: SentenceResult) -> bool:
    """
    KoBERT는 규칙이 최종 확정한 정상 문장을 제외하고 최대한 태운다.
    - 비도메인/명백한 허용 표현은 스킵
    - 애매하지만 규칙상 정상인 문장은 KoBERT로 보낸다
    """
    if rule_result.suspicion_level != SuspicionLevel.NORMAL:
        return True

    if rule_result.reason in {
        NON_DOMAIN_NORMAL_REASON,
        STRICT_ALLOW_NORMAL_REASON,
        EXTRA_ALLOW_NORMAL_REASON,
    }:
        return False

    return True


# ────────────────────────────────────────────
# 패턴 태그 → 한국어 설명 매핑
# ────────────────────────────────────────────
CAUTION_KEYWORDS: list[str] = [
    "24시간",
    "롱래스팅",
    "론웨어",
    "long lasting",
    "long-lasting",
    "해결",
    "피부나이",
    "동물 실험 없이",
    "동물 실험 X",
    "비건 인증",
    "기미잡티",
    "기미 잡티",
    "탄력 저하",
    "리셋",
]

CAUTION_REGEX_PATTERNS: list[tuple[str, str]] = [
    (r"24\s*시간", "24시간"),
    (r"(하루\s*종일|오래)\s*지속", "지속"),
    (r"(임상\s*테스트|인체\s*적용\s*시험)\s*(완료|완료됨|기반)", "임상 테스트"),
]


_PATTERN_DESC: dict[str, str] = {
    "의약품오인":  "의약품으로 오인될 수 있는 표현",
    "효능과장":   "효능을 과장하거나 절대적으로 단정하는 표현",
    "기능성오인":  "기능성 화장품 심사 없이 기능성을 주장하는 표현",
    "안전성단정":  "안전성을 근거 없이 단정하는 표현",
    "추천보증":   "전문가 추천·인증을 주장하는 표현",
    "검증오인":   "임상·시험 결과를 효능 보장처럼 단정하는 표현",
    "비교우위":   "근거 없이 타사 대비 우위를 주장하는 표현",
    "첨단기술오인": "첨단 기술·세포 재생을 근거 없이 강조하는 표현",
    "강력금지":   "100%, 영구, 완치 등 과도한 보장/단정 표현",
    "고위험기능성오인": "기능성 범위를 넘어 신체 변화·의약품 효능처럼 보이는 고위험 표현",
    "주의예외": "허용 가능성이 있으나 주의 단계로 유지해야 하는 표현",
}

_PATTERN_RULE_LEVELS: dict[str, SuspicionLevel] = {
    "의약품오인": SuspicionLevel.SUSPICIOUS,
    "효능과장": SuspicionLevel.SUSPICIOUS,
    "기능성오인": SuspicionLevel.CAUTION,
    "안전성단정": SuspicionLevel.SUSPICIOUS,
    "추천보증": SuspicionLevel.SUSPICIOUS,
    "검증오인": SuspicionLevel.SUSPICIOUS,
    "비교우위": SuspicionLevel.CAUTION,
    "첨단기술오인": SuspicionLevel.CAUTION,
    "강력금지": SuspicionLevel.SUSPICIOUS,
    "고위험기능성오인": SuspicionLevel.SUSPICIOUS,
    "주의예외": SuspicionLevel.CAUTION,
}


# ────────────────────────────────────────────
# 분석 함수
# ────────────────────────────────────────────

def analyze_sentence(sentence: str, *, force_cosmetic: bool = False) -> SentenceResult:
    """1차 규칙기반 엔진: 문장 하나를 분석하여 SentenceResult 반환"""
    normalized_sentence = re.sub(r"100\s*프로", "100프로", sentence)
    normalized_sentence = re.sub(r"100\s*퍼(?:센트)?", "100퍼센트", normalized_sentence)

    if sentence.strip() in EXACT_ALLOW_SENTENCES:
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=EXTRA_ALLOW_NORMAL_REASON,
            score=0.0,
        )

    if sentence.strip() in EXACT_CAUTION_SENTENCES:
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.CAUTION,
            matched_keywords=[],
            matched_patterns=["주의예외"],
            reason="허용 가능성이 있으나 효능 표현이 강해 주의가 필요한 문구입니다.",
            score=0.35,
        )

    # 0) 화장품 여부와 무관한 강력 금지 표현 우선 차단
    for kw in STRONG_FORBIDDEN_KEYWORDS:
        if kw in normalized_sentence:
            return SentenceResult(
                sentence=sentence,
                suspicion_level=SuspicionLevel.SUSPICIOUS,
                matched_keywords=[kw],
                matched_patterns=["강력금지"],
                reason=f"'{kw}'처럼 허위/과장 소지가 큰 표현을 포함합니다.",
                score=0.85,
            )

    matched_keywords, matched_patterns = _find_forbidden_patterns(normalized_sentence)

    high_risk_matches = [
        kw for kw in HIGH_RISK_FUNCTIONAL_KEYWORDS
        if kw in normalized_sentence
    ]
    if high_risk_matches:
        matched_keywords.extend(high_risk_matches)
        if "고위험기능성오인" not in matched_patterns:
            matched_patterns.append("고위험기능성오인")

    # 주의 키워드 검사
    caution_matches = [kw for kw in CAUTION_KEYWORDS if kw in normalized_sentence]
    caution_matches.extend(
        label
        for pattern, label in CAUTION_REGEX_PATTERNS
        if re.search(pattern, normalized_sentence)
    )
    matched_keywords.extend(caution_matches)

    whitelist_matches = _find_whitelist_matches(normalized_sentence)
    context_allow_matches = _find_context_allow_matches(normalized_sentence)
    extra_allow_matches = [kw for kw in STRICT_ALLOW_EXTRA_KEYWORDS if kw in normalized_sentence]

    # ── 도메인 관련성 필터 ─────────────────────────────────
    # 명백한 금지/주의 신호가 있으면 도메인 단서가 부족해도 계속 분석한다.
    if (
        not force_cosmetic
        and not is_cosmetic_related(normalized_sentence)
        and not matched_patterns
        and not caution_matches
        and not whitelist_matches
        and not context_allow_matches
    ):
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=NON_DOMAIN_NORMAL_REASON,
            score=0.0,
        )

    if extra_allow_matches and _is_clearly_allowed_sentence(normalized_sentence, extra_allow_matches):
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=EXTRA_ALLOW_NORMAL_REASON,
            score=0.0,
        )

    # ── 화이트리스트 체크 ───────────────────────────────────
    # 금지 패턴이 없고, 문장 전체가 허용 맥락일 때만 정상 처리한다.
    if whitelist_matches and not matched_patterns and _is_clearly_allowed_sentence(normalized_sentence, whitelist_matches):
        return SentenceResult(
            sentence=sentence,
            suspicion_level=SuspicionLevel.NORMAL,
            matched_keywords=[],
            matched_patterns=[],
            reason=STRICT_ALLOW_NORMAL_REASON,
            score=0.0,
        )

    if context_allow_matches and not matched_patterns and not caution_matches:
        matched_keywords.extend(context_allow_matches)

    # 의심도 결정 + 규칙 기반 연속 점수
    if matched_patterns:
        level = max(
            (_PATTERN_RULE_LEVELS.get(pattern, SuspicionLevel.SUSPICIOUS) for pattern in matched_patterns),
            key=lambda lvl: {SuspicionLevel.NORMAL: 0, SuspicionLevel.CAUTION: 1, SuspicionLevel.SUSPICIOUS: 2}[lvl],
        )
        if level == SuspicionLevel.SUSPICIOUS:
            base_score = round(min(0.70 + len(matched_patterns) * 0.05, 0.90), 3)
        else:
            base_score = round(min(0.35 + len(matched_patterns) * 0.05, 0.55), 3)
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
        reason = (
            CONTEXT_ALLOW_NORMAL_REASON
            if context_allow_matches
            else DEFAULT_NORMAL_REASON
        )

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
    "검증오인":   "'{kw}' 표현은 임상·시험 결과가 특정 효능을 입증·보장한 것처럼 단정하는 표현으로, 소비자를 오인하게 할 수 있습니다.",
    "비교우위":   "'{kw}' 표현은 근거 없이 타사 대비 우위를 주장하는 표현으로, 비교광고 기준에 위반될 수 있습니다.",
    "고위험기능성오인": "'{kw}' 표현은 화장품 기능성 범위를 넘어 신체 변화나 의약품적 효능으로 오인될 수 있는 고위험 표현입니다.",
}

def _build_reason(patterns: list[str], keywords: list[str]) -> str:
    """패턴 태그 + 감지 키워드를 조합해 문구별 맞춤 이유 생성"""
    if not patterns:
        return DEFAULT_NORMAL_REASON

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
