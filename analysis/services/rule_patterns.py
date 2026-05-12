from analysis.models.schemas import SuspicionLevel


PATTERN_DESCRIPTIONS: dict[str, str] = {
    "의약품오인": "의약품으로 오인될 수 있는 표현",
    "효능과장": "효능을 과장하거나 절대적으로 단정하는 표현",
    "기능성오인": "기능성 화장품 심사 없이 기능성을 주장하는 표현",
    "안전성단정": "안전성을 근거 없이 단정하는 표현",
    "추천보증": "전문가 추천·인증을 주장하는 표현",
    "검증오인": "임상·시험 결과를 효능 보장처럼 단정하는 표현",
    "비교우위": "근거 없이 타사 대비 우위를 주장하는 표현",
    "첨단기술오인": "첨단 기술·세포 재생을 근거 없이 강조하는 표현",
    "강력금지": "100%, 영구, 완치 등 과도한 보장/단정 표현",
    "고위험기능성오인": "기능성 범위를 넘어 신체 변화·의약품 효능처럼 보이는 고위험 표현",
    "주의예외": "허용 가능성이 있으나 주의 단계로 유지해야 하는 표현",
}


PATTERN_RULE_LEVELS: dict[str, SuspicionLevel] = {
    "의약품오인": SuspicionLevel.SUSPICIOUS,
    "효능과장": SuspicionLevel.CAUTION,
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


PATTERN_REASON_TEMPLATES: dict[str, str] = {
    "의약품오인": "'{kw}'처럼 치료나 예방 효과로 받아들여질 수 있는 표현이 포함되어 있습니다.",
    "효능과장": "'{kw}'처럼 효과를 지나치게 단정하거나 과장한 표현이 포함되어 있습니다.",
    "기능성오인": "'{kw}'처럼 기능성을 강하게 내세우는 표현이 포함되어 있습니다.",
    "안전성단정": "'{kw}'처럼 안전성을 단정하는 표현이 포함되어 있습니다.",
    "추천보증": "'{kw}'처럼 전문가 추천이나 인증으로 받아들여질 수 있는 표현이 포함되어 있습니다.",
    "검증오인": "'{kw}'처럼 시험이나 검증 결과를 단정적으로 전달하는 표현이 포함되어 있습니다.",
    "비교우위": "'{kw}'처럼 다른 제품보다 우수하다고 받아들여질 수 있는 표현이 포함되어 있습니다.",
    "첨단기술오인": "'{kw}'처럼 첨단 기술이나 재생 효과를 강조하는 표현이 포함되어 있습니다.",
    "강력금지": "'{kw}'처럼 지나치게 단정적인 표현이 포함되어 있습니다.",
    "고위험기능성오인": "'{kw}'처럼 신체 변화나 강한 효능으로 받아들여질 수 있는 표현이 포함되어 있습니다.",
    "주의예외": "'{kw}' 표현은 허용 가능성이 있지만, 표현 강도가 높아 주의가 필요합니다.",
}


MIN_PATTERN_LEVELS = PATTERN_RULE_LEVELS

CAUTION_ONLY_PATTERNS = {"기능성오인", "첨단기술오인", "비교우위", "주의예외"}

STRONG_SUSPICIOUS_PATTERNS = {
    "강력금지",
    "의약품오인",
    "안전성단정",
    "추천보증",
    "검증오인",
    "고위험기능성오인",
}
