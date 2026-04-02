"""
화장품 허위·과장 광고 문구 데이터 증강 스크립트 (CSV 패턴 기반 + 완성 문장 위주)
딱 걸렸어! 프로젝트

전략:
  - CSV에서 추출한 실제 키워드·표현구를 문장 템플릿에 끼워 완성 문장 생성 (70%)
  - 기존 CSV 실제 문장들을 어그멘테이션으로 변형 (30%)

- 입력: data/merged/merged_clean.csv
- 출력: data/merged/merged_clean.csv (덮어쓰기)
- 목표: 총 10000행

실행:
    python data/augment_data.py
    python data/augment_data.py --target 10000
"""

import os
import re
import random
import argparse
import pandas as pd

random.seed(42)

# ────────────────────────────────────────────
# 경로
# ────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_PATH  = os.path.join(PROJECT_ROOT, "data", "merged", "merged_clean.csv")
CLEAN_DIR    = os.path.join(PROJECT_ROOT, "data", "clean")

TARGET_TOTAL = 18000
LABEL_RATIO  = {"의심": 0.50, "주의": 0.35, "정상": 0.15}

COLUMNS = [
    "정제 ID", "원본 ID", "정제 문구", "키워드",
    "패턴 태그", "기준 라벨", "의심도 라벨", "사유",
]


# ════════════════════════════════════════════
# CSV에서 추출한 실제 키워드·표현구 사전
# (clean_*.csv 파일들의 정제 문구를 직접 참고하여 구성)
# ════════════════════════════════════════════

# ── [의심] 키워드 및 표현구 ─────────────────
SUSPICIOUS_KEYWORDS = {
    # 의약품 오인
    "의약품오인": [
        "아토피", "건선", "습진", "소양증", "모낭충", "기저귀 발진",
        "여드름", "두드러기", "주부습진", "지루성 피부염", "접촉성 피부염",
        "살균·소독", "살균소독", "항균", "항염·진통", "항진균·항바이러스",
        "해독", "이뇨", "항암", "근육 이완", "통증 경감", "면역 강화",
        "혈액순환", "피부재생", "세포재생", "호르몬 분비 촉진",
        "유익균 균형 보호", "질염 예방", "세포성장 촉진", "세포 활력",
        "DNA 활성화", "심신피로 회복", "찰과상·화상 치료",
        "탈모 방지·치료", "발모·육모·양모", "모발 성장 촉진",
        "코스메슈티컬", "메디슨", "드럭",
    ],
    # 효능 과장·절대 표현
    "효능과장": [
        "100% 효과 보장", "즉각 세균 99.9% 억제율",
        "모발 생성이 426% 증가", "모발 굵기 증가",
        "홍티의 흔적을 없애준다", "얼굴 피부 고민은 여기서 끝내버리세요",
        "부작용이 전혀 없다", "먹을 수 있다",
        "소독 효과는 의료용 에탄올과 동일", "강력한 염증 진정 효과",
        "피하지방 분해", "지방분해를 자극하고 지방세포를 감소",
        "바르기만 하면 열이 소모돼 살이 타게 되고 지방분해에 도움",
        "특허성분으로 진피층까지 강력투입",
    ],
    # 기능성 오인
    "기능성오인": [
        "기미·주근깨", "셀룰라이트", "붓기·다크서클", "임신선·튼살",
        "피부 독소 제거", "홍조·홍반 개선", "뾰루지 개선",
        "피부노화", "가슴 탄력·확대", "얼굴 크기 축소",
        "얼굴 윤곽·V라인", "체형·몸매 변화", "다이어트",
        "보톡스", "필러", "바르는 Botox", "기미톡스",
        "쥐젖 제거", "줄기세포가 들어 있는 것으로 오인",
        "지방볼륨생성", "모낭 주기 조절",
    ],
    # 추천·보증
    "추천보증": [
        "OO 아토피 협회 인증 화장품", "OO 의료기관의 첨단기술로 탄생한 화장품",
        "의사가 개발한 화장품", "OO 병원에서 추천하는 화장품",
        "식약처 허가·인증 받은 제품", "피부과 테스트 완료",
        "OO시험기관의 OO 효과 입증",
    ],
}

# ── [주의] 키워드 및 표현구 ─────────────────
CAUTION_KEYWORDS = {
    "효능과장": [
        "탄력 강화", "모발 두께 증가", "두피 진피층까지 영양 전달",
        "안티에이징", "피부노화 완화", "피부노화 징후 감소",
        "콜라겐 증가·활성화", "피부 혈행 개선", "피부장벽 손상의 개선에 도움",
        "피부 피지분비 조절", "미세먼지 차단", "빠지는 모발 감소",
        "수분감 OO% 개선효과", "피부결 OO% 개선",
    ],
    "기능성오인": [
        "주름 개선에 도움", "기미·주근깨 완화에 도움",
        "일시적 셀룰라이트 감소", "붓기 완화", "다크서클 완화",
        "모발 손상 개선", "여드름성 피부 사용에 적합",
    ],
    "체험담유도": [
        "홍티가 사라졌네요", "미백효과·피부질감·눈가 주름이 개선되었습니다",
        "사용 후 피부가 달라졌어요", "직접 써보니 효과 있어요",
    ],
}

# ── [정상] 키워드 및 표현구 ─────────────────
NORMAL_KEYWORDS = {
    "허용표현": [
        "피부를 건강하게 가꾸는 데 도움", "피부 보습에 도움",
        "피부에 생기를 부여", "수분 공급에 도움",
        "피부 결 정돈에 도움", "피부 진정에 도움",
        "촉촉한 사용감", "산뜻한 마무리감",
        "식약처 허가 기능성 화장품 (미백에 도움)",
        "식약처 허가 기능성 화장품 (주름 개선에 도움)",
        "자외선 차단 기능 (SPF50+/PA++++)",
        "천연(유래)지수 OO% (ISO 16128 계산 적용)",
        "피부과 테스트 완료", "모발 손상 개선에 도움",
        "빠지는 모발 감소에 도움",
    ],
}

# ════════════════════════════════════════════
# 문장 템플릿
# keyword 자리에 위 사전의 실제 표현이 들어감
# ════════════════════════════════════════════

# ── [의심] 완성 문장 템플릿 ──────────────────
SUSPICIOUS_SENTENCE_TEMPLATES = [
    # 의약품 오인
    ("이 크림은 {kw}에 탁월한 효과가 있습니다.",          "의약품오인,효능과장",  "0"),
    ("{kw} 치료에 직접적인 도움을 드립니다.",              "의약품오인,질병치료표현", "0"),
    ("{kw} 증상이 있으신 분께 강력 추천드립니다.",         "의약품오인,질병치료표현", "0"),
    ("피부과 전문의도 인정한 {kw} 케어 화장품.",           "의약품오인,추천보증",   "0"),
    ("{kw} 걱정, 이 제품 하나로 해결하세요.",              "의약품오인,효능과장",   "0"),
    ("임상 시험으로 {kw} 개선 효과가 입증되었습니다.",     "의약품오인,추천보증",   "0"),
    ("의사가 직접 개발한 {kw} 전용 크림.",                 "의약품오인,추천보증",   "0"),
    ("병원에서 권장하는 {kw} 솔루션.",                     "의약품오인,추천보증",   "0"),
    # 효능 과장
    ("단 7일 만에 {kw}를 직접 확인하세요!",                "효능과장,절대표현",     "0"),
    ("{kw}! 사용 즉시 효과를 느낄 수 있습니다.",           "효능과장,절대표현",     "0"),
    ("100% 효과 보장! {kw} 화장품.",                       "효능과장,절대표현",     "0"),
    ("부작용 없이 {kw}를 경험하세요.",                     "안전성단정,효능과장",   "0"),
    ("모든 피부 타입에 안전한 {kw} 제품.",                 "안전성단정,절대표현",   "0"),
    ("{kw} 세계 최초 특허 기술 적용.",                     "효능과장,추천보증",     "0"),
    ("국내 유일! {kw} 완벽 실현.",                         "효능과장,절대표현",     "0"),
    # 기능성 오인
    ("단 14일 만에 {kw}을 완전히 없애드립니다.",           "기능성오인,절대표현",   "0"),
    ("{kw} 고민, 이제 이 제품으로 끝내세요.",              "기능성오인,효능과장",   "0"),
    ("{kw}를 뿌리부터 해결하는 기능성 크림.",              "기능성오인,효능과장",   "0"),
    ("시술 없이 {kw} 효과! 바르기만 하면 됩니다.",         "기능성오인,절대표현",   "0"),
    # 추천·인증
    ("{kw}으로 유명한 병원에서 공식 인증한 제품.",         "추천보증,의약품오인",   "0"),
    ("전문의 {kw} 검증 완료. 믿고 사용하세요.",            "추천보증",              "0"),
    ("{kw} 인증마크 획득! 검증된 효과.",                   "추천보증,효능과장",     "0"),
]

# ── [주의] 완성 문장 템플릿 ──────────────────
CAUTION_SENTENCE_TEMPLATES = [
    ("{kw} 효과로 더 자신감 있는 피부를 만들어보세요.",    "효능과장",              "NA"),
    ("혁신적인 기술로 {kw}를 실현했습니다.",               "효능과장",              "NA"),
    ("빠른 시간 안에 {kw}를 직접 느껴보세요.",             "효능과장",              "NA"),
    ("과학적으로 입증된 {kw} 솔루션.",                     "추천보증,효능과장",     "NA"),
    ("차세대 기술 적용으로 {kw} 극대화.",                  "효능과장",              "NA"),
    ("{kw} 변화, 사용 후 직접 확인해보세요.",              "효능과장,체험담유도",   "NA"),
    ("피부 나이를 되돌리는 {kw} 케어.",                    "효능과장,기능성오인",   "NA"),
    ("{kw}에 특화된 고농축 앰플.",                         "효능과장",              "NA"),
    ("놀라운 {kw} 변화를 경험하세요.",                     "효능과장",              "NA"),
    ("획기적인 성분으로 {kw} 실현.",                       "효능과장",              "NA"),
    ("{kw}, 이제는 집에서도 가능합니다.",                  "효능과장",              "NA"),
    ("특허받은 기술로 {kw}를 도와드립니다.",               "추천보증,효능과장",     "NA"),
    ("전문가가 인정한 {kw} 효과.",                         "추천보증,효능과장",     "NA"),
    ("{kw} 개선, 임상 결과로 확인된 제품.",                "추천보증,효능과장",     "NA"),
]

# ── [정상] 완성 문장 템플릿 ──────────────────
NORMAL_SENTENCE_TEMPLATES = [
    ("{kw}을 위한 일상 피부 케어 제품.",                   "일반홍보,허용표현",     "1"),
    ("가볍고 산뜻하게 {kw}을 도와주는 크림.",              "허용표현,일반홍보",     "1"),
    ("{kw}에 도움을 주는 성분 함유 에센스.",               "허용표현,일반홍보",     "1"),
    ("피부를 촉촉하게 가꾸어주는 {kw} 제품.",              "일반홍보,허용표현",     "1"),
    ("매일 사용하기 좋은 {kw} 케어 로션.",                 "일반홍보,허용표현",     "1"),
    ("{kw}을 위한 부드러운 텍스처의 세럼.",                "허용표현,일반홍보",     "1"),
    ("하루 종일 지속되는 {kw} 효과.",                      "일반홍보,허용표현",     "1"),
    ("식약처 허가 성분으로 구성된 {kw} 화장품.",           "허용표현",              "1"),
]

# ════════════════════════════════════════════
# 어그멘테이션 (기존 문장 변형)
# ════════════════════════════════════════════

SYNONYM_MAP = {
    "피부": ["피부결", "스킨"],
    "개선": ["향상", "케어"],
    "효과": ["효능", "결과"],
    "크림": ["로션", "세럼", "앰플"],
    "도움": ["도움을 줌", "효과"],
    "제거": ["완화", "케어"],
    "치료": ["개선", "완화"],
    "보습": ["수분 공급", "수분 유지"],
    "추천": ["권장", "제안"],
    "강력한": ["탁월한", "뛰어난"],
    "촉촉": ["수분감 있는", "촉촉하게"],
}

ENDING_VARIANTS = [
    ("합니다", "해요"),
    ("합니다", "합니다!"),
    ("입니다", "이에요"),
    ("입니다", "입니다!"),
    ("하세요", "해보세요"),
    ("줍니다", "드립니다"),
    ("됩니다", "돼요"),
    ("습니다", "어요"),
    ("세요", "십시오"),
]

PREFIXES = [
    "피부과학 기반 ", "더마 ", "프리미엄 ",
    "집중 케어 ", "고농축 ", "순한 ",
]


def augment_by_synonym(text: str) -> str | None:
    for word, synonyms in SYNONYM_MAP.items():
        if word in text:
            return text.replace(word, random.choice(synonyms), 1)
    return None


def augment_by_ending(text: str) -> str | None:
    for original, variant in random.sample(ENDING_VARIANTS, len(ENDING_VARIANTS)):
        if original in text:
            return text.replace(original, variant, 1)
    return None


def augment_by_prefix(text: str) -> str:
    return random.choice(PREFIXES) + text


# ════════════════════════════════════════════
# 정규화 (중복 체크용)
# ════════════════════════════════════════════

def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


# ════════════════════════════════════════════
# ID 생성
# ════════════════════════════════════════════

def make_aug_id(seq: int) -> tuple[str, str]:
    raw_id = f"RAW-AUG-000-{seq:05d}"
    return f"CLN-{raw_id}", raw_id


# ════════════════════════════════════════════
# 문장 생성 (키워드 → 완성 문장)
# ════════════════════════════════════════════

def generate_sentences(
    templates: list[tuple],
    keyword_dict: dict,
    count: int,
    existing_normalized: set,
) -> list[tuple[str, str, str]]:
    """템플릿 + 실제 키워드 조합으로 완성 문장 생성"""
    results = []
    all_keywords = [(kw, ptag) for ptag, kws in keyword_dict.items() for kw in kws]
    attempts = 0

    while len(results) < count and attempts < count * 20:
        attempts += 1
        template, pattern_tag, base_label = random.choice(templates)
        kw, kw_ptag = random.choice(all_keywords)

        phrase = template.replace("{kw}", kw)
        norm = normalize(phrase)

        if norm not in existing_normalized and len(phrase) > 5:
            # 패턴 태그: 템플릿 태그 + 키워드 태그 합치기 (중복 제거)
            tags = list(dict.fromkeys(
                pattern_tag.split(",") + [kw_ptag]
            ))
            results.append((phrase, ",".join(tags), base_label))
            existing_normalized.add(norm)

    return results


def generate_augmented(sources: list[str], count: int, existing_normalized: set) -> list[str]:
    """기존 문장 어그멘테이션"""
    results = []
    attempts = 0

    while len(results) < count and attempts < count * 10:
        attempts += 1
        if not sources:
            break
        base = random.choice(sources)
        method = random.choice(["synonym", "ending", "prefix"])

        if method == "synonym":
            new = augment_by_synonym(base)
        elif method == "ending":
            new = augment_by_ending(base)
        else:
            new = augment_by_prefix(base)

        if new and new != base:
            norm = normalize(new)
            if norm not in existing_normalized:
                results.append(new)
                existing_normalized.add(norm)

    return results


# ════════════════════════════════════════════
# 라벨별 설정
# ════════════════════════════════════════════

LABEL_CONFIG = {
    "의심": {
        "sentence_templates": SUSPICIOUS_SENTENCE_TEMPLATES,
        "keyword_dict":        SUSPICIOUS_KEYWORDS,
        "base_label":          "0",
        "pattern_default":     "효능과장",
        "reason":              "CSV 패턴 기반 생성 — 허위·과장 광고 의심 표현",
    },
    "주의": {
        "sentence_templates": CAUTION_SENTENCE_TEMPLATES,
        "keyword_dict":        CAUTION_KEYWORDS,
        "base_label":          "NA",
        "pattern_default":     "효능과장",
        "reason":              "CSV 패턴 기반 생성 — 과장 가능성 있는 표현",
    },
    "정상": {
        "sentence_templates": NORMAL_SENTENCE_TEMPLATES,
        "keyword_dict":        NORMAL_KEYWORDS,
        "base_label":          "1",
        "pattern_default":     "허용표현",
        "reason":              "CSV 패턴 기반 생성 — 허용 범위 내 표현",
    },
}


# ════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════

def augment(target_total: int = TARGET_TOTAL):
    print("\n" + "=" * 55)
    print("📈 데이터 증강 시작 (CSV 패턴 기반 + 완성 문장 위주)")
    print("=" * 55)

    # 기존 데이터 로드
    if os.path.exists(MERGED_PATH):
        df_existing = pd.read_csv(MERGED_PATH, encoding="utf-8-sig", dtype=str)
        existing_count = len(df_existing)
        existing_phrases = set(df_existing["정제 문구"].dropna().tolist()) \
            if "정제 문구" in df_existing.columns else set()
    else:
        print("⚠️  병합 파일 없음 — 빈 데이터에서 시작합니다.")
        df_existing     = pd.DataFrame(columns=COLUMNS)
        existing_count  = 0
        existing_phrases = set()

    existing_normalized = {normalize(p) for p in existing_phrases}

    need_count = max(0, target_total - existing_count)
    print(f"\n📊 현재 데이터: {existing_count}행")
    print(f"🎯 목표 데이터: {target_total}행")
    print(f"➕ 생성 필요:   {need_count}행\n")

    if need_count <= 0:
        print("✅ 이미 목표 행 수를 달성했습니다.")
        return

    # 라벨별 생성 수
    label_counts = {l: int(need_count * r) for l, r in LABEL_RATIO.items()}
    label_counts["의심"] += need_count - sum(label_counts.values())

    print("📋 라벨별 생성 계획:")
    for label, cnt in label_counts.items():
        print(f"   {label}: {cnt}개  (완성문장 {int(cnt*0.7)}개 + 어그멘테이션 {cnt-int(cnt*0.7)}개)")
    print()

    all_new_rows = []
    seq = existing_count + 1

    for label, total in label_counts.items():
        if total <= 0:
            continue

        cfg = LABEL_CONFIG[label]
        sentence_count = int(total * 0.7)
        aug_count      = total - sentence_count

        print(f"🔄 [{label}] 생성 중...")

        # ── ① 완성 문장 생성 (키워드 → 템플릿)
        sentence_results = generate_sentences(
            cfg["sentence_templates"],
            cfg["keyword_dict"],
            sentence_count,
            existing_normalized,
        )

        for phrase, pattern_tag, base_label in sentence_results:
            cln_id, raw_id = make_aug_id(seq)
            kws = [w for w in re.findall(r"[가-힣a-zA-Z·%]+", phrase) if len(w) > 1][:3]
            all_new_rows.append({
                "정제 ID":    cln_id,
                "원본 ID":    raw_id,
                "정제 문구":  phrase,
                "키워드":     ",".join(kws),
                "패턴 태그":  pattern_tag,
                "기준 라벨":  base_label,
                "의심도 라벨": label,
                "사유":       cfg["reason"],
            })
            seq += 1

        # ── ② 어그멘테이션 (기존 실제 문장 변형)
        # 해당 라벨의 기존 문장 중 10자 이상만 어그멘테이션 소스로 사용
        if "의심도 라벨" in df_existing.columns and "정제 문구" in df_existing.columns:
            sources = df_existing[
                (df_existing["의심도 라벨"] == label) &
                (df_existing["정제 문구"].str.len() >= 10)
            ]["정제 문구"].dropna().tolist()
        else:
            sources = []

        aug_phrases = generate_augmented(sources, aug_count, existing_normalized)

        for phrase in aug_phrases:
            cln_id, raw_id = make_aug_id(seq)
            kws = [w for w in re.findall(r"[가-힣a-zA-Z·%]+", phrase) if len(w) > 1][:3]
            all_new_rows.append({
                "정제 ID":    cln_id,
                "원본 ID":    raw_id,
                "정제 문구":  phrase,
                "키워드":     ",".join(kws),
                "패턴 태그":  cfg["pattern_default"],
                "기준 라벨":  cfg["base_label"],
                "의심도 라벨": label,
                "사유":       cfg["reason"] + " (어그멘테이션)",
            })
            seq += 1

        label_total = sum(1 for r in all_new_rows if r["의심도 라벨"] == label)
        print(f"   └ [{label}] {label_total}개 완료")

    # ── 저장
    if not all_new_rows:
        print("\n❌ 생성된 데이터가 없습니다.")
        return

    df_new   = pd.DataFrame(all_new_rows, columns=COLUMNS)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final = df_final.fillna("NA").replace("", "NA")

    os.makedirs(os.path.dirname(MERGED_PATH), exist_ok=True)
    df_final.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 55}")
    print(f"✅ 증강 완료!")
    print(f"   기존: {existing_count}행")
    print(f"   추가: {len(all_new_rows)}행")
    print(f"   최종: {len(df_final)}행")
    print(f"   저장: {MERGED_PATH}")
    print(f"{'=' * 55}\n")


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="데이터 증강 (CSV 패턴 기반) — 딱 걸렸어! 프로젝트"
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_TOTAL,
        help=f"목표 총 행 수 (기본값: {TARGET_TOTAL})"
    )
    args = parser.parse_args()
    augment(target_total=args.target)
