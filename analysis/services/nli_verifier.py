"""
3차 NLI 기반 클레임 검증 모듈

호출 조건 (ai_analyzer.py에서 제어):
  - 규칙 엔진이 패턴을 감지했고 (matched_patterns 존재)
  - KoBERT 점수가 회색지대(0.35~0.65)이거나 규칙 엔진·KoBERT 결과가 엇갈릴 때

동작:
  - 감지된 패턴 태그에 해당하는 위반 가설만 선택적으로 검증 (속도 최적화)
  - entailment 높음 → 위반 확인, 점수 상향 또는 레벨 업그레이드
  - contradiction 높음 → 위반 아님, 점수 하향
  - 중립 → KoBERT 결과 유지

모델: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
  - XNLI 학습으로 한국어 포함 15개 언어 지원
  - 별도 fine-tuning 없이 바로 사용 가능
  - 라벨 순서: 0=entailment, 1=neutral, 2=contradiction
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analysis.models.schemas import SentenceResult, SuspicionLevel

NLI_MODEL_NAME = os.getenv("NLI_MODEL_PATH", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTAILMENT_IDX    = 0
CONTRADICTION_IDX = 2

NLI_CONFIRM_THRESHOLD = 0.60  # entailment >= 이 값이면 위반 가설 확인
NLI_DENY_THRESHOLD    = 0.50  # contradiction >= 이 값이면 위반 부정

_nli_tokenizer = None
_nli_model     = None


def _load_nli_model() -> bool:
    global _nli_tokenizer, _nli_model
    if _nli_model is not None:
        return True
    try:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
        _nli_model.eval()
        print(f"[NLI] 모델 로드 완료: {NLI_MODEL_NAME}")
        return True
    except Exception as e:
        print(f"[NLI] 모델 로드 실패: {e}")
        return False


# ── 위반 가설 목록 ────────────────────────────────────────────
# 규칙 엔진 패턴 태그 → [(가설 텍스트, 위반 시 부여 레벨)]
# 패턴 태그별로 분리해서 관련 가설만 선택적으로 검증

VIOLATION_HYPOTHESES: dict[str, list[tuple[str, SuspicionLevel]]] = {

    "의약품오인": [
        ("이 문장은 화장품이 질병을 치료하거나 예방한다고 주장한다",              SuspicionLevel.SUSPICIOUS),
        ("이 문장은 화장품을 의약품처럼 의학적 효능이 있다고 표현한다",            SuspicionLevel.SUSPICIOUS),
        ("이 문장은 피부 염증·감염·알레르기를 치료할 수 있다고 주장한다",          SuspicionLevel.SUSPICIOUS),
    ],

    "효능과장": [
        ("이 문장은 화장품 효과를 100% 확실하게 보장한다고 주장한다",             SuspicionLevel.SUSPICIOUS),
        ("이 문장은 짧은 기간 안에 눈에 띄는 효과가 반드시 난다고 단정한다",       SuspicionLevel.SUSPICIOUS),
        ("이 문장은 화장품이 완벽하거나 절대적인 효과를 낸다고 주장한다",          SuspicionLevel.SUSPICIOUS),
    ],

    "기능성오인": [
        ("이 문장은 허가 없이 주름 개선·미백·자외선 차단 기능을 주장한다",         SuspicionLevel.CAUTION),
        ("이 문장은 피부 세포나 모발을 재생시킬 수 있다고 주장한다",               SuspicionLevel.CAUTION),
        ("이 문장은 체중 감소나 체형 변화 효과가 있다고 주장한다",                 SuspicionLevel.CAUTION),
    ],

    "안전성단정": [
        ("이 문장은 이 제품에 부작용이 전혀 없다고 단정한다",                      SuspicionLevel.SUSPICIOUS),
        ("이 문장은 모든 사람의 피부에 완전히 안전하다고 주장한다",                 SuspicionLevel.SUSPICIOUS),
        ("이 문장은 자극이나 알레르기 반응이 절대 없다고 보장한다",                 SuspicionLevel.SUSPICIOUS),
    ],

    "추천보증": [
        ("이 문장은 의사나 의료 전문가가 이 제품을 추천하거나 인증했다고 주장한다", SuspicionLevel.SUSPICIOUS),
        ("이 문장은 공인 기관의 인증이나 허가를 받았다고 주장한다",                SuspicionLevel.SUSPICIOUS),
    ],

    "비교우위": [
        ("이 문장은 다른 제품보다 이 제품이 더 우수하다고 주장한다",               SuspicionLevel.CAUTION),
        ("이 문장은 이 제품이 업계에서 유일하거나 최고라고 주장한다",               SuspicionLevel.CAUTION),
    ],

    "첨단기술오인": [
        ("이 문장은 줄기세포·엑소좀 등 첨단 기술로 피부를 재생한다고 주장한다",    SuspicionLevel.CAUTION),
        ("이 문장은 과학적으로 검증되지 않은 피부 재생 기술 효과를 주장한다",      SuspicionLevel.CAUTION),
    ],
}

_LEVEL_ORDER = {
    SuspicionLevel.NORMAL:     0,
    SuspicionLevel.CAUTION:    1,
    SuspicionLevel.SUSPICIOUS: 2,
}


# ── 화장품 도메인 판별 (Zero-shot) ───────────────────────────
# contradiction이 높으면 화장품 광고가 아닌 것으로 판단 (더 안정적)
COSMETIC_HYPOTHESES = [
    "이 글은 화장품 또는 뷰티 제품에 관한 내용이다",
    "이 글은 피부, 모발, 또는 미용 관련 제품을 다루고 있다",
    "이 글은 스킨케어, 메이크업, 또는 헤어케어 제품을 소개한다",
    "이 글은 크림, 세럼, 로션, 앰플 등 화장품을 광고한다",
    "이 글은 피부 보습, 미백, 주름 개선 등 뷰티 효과를 설명한다",
]
NON_COSMETIC_HYPOTHESES = [
    "이 글은 전자제품, 가전제품, 또는 식품에 관한 내용이다",
    "이 글은 화장품과 무관한 제품이나 서비스를 다루고 있다",
    "이 글은 음식, 식재료, 또는 요리에 관한 내용이다",
    "이 글은 의류, 신발, 또는 패션 잡화에 관한 내용이다",
    "이 글은 가구, 생활용품, 또는 인테리어에 관한 내용이다",
    "이 글은 의약품, 건강기능식품, 또는 의료기기에 관한 내용이다",
]
COSMETIC_CONTRADICTION_THRESHOLD = 0.45


def is_cosmetic_ad(text: str) -> bool:
    """Zero-shot NLI로 화장품·뷰티 광고 여부 판별.
    모든 가설에 대해 contradiction이 높으면 화장품 아님으로 판단."""
    if not _load_nli_model():
        return True  # 모델 로드 실패 시 분석 계속 진행
    try:
        preview = text[:30].replace("\n", " ")
        # 화장품 가설 최대 entailment
        best_cosmetic_entail = 0.0
        for i, hypothesis in enumerate(COSMETIC_HYPOTHESES):
            entail, contra = _get_nli_probs(text, hypothesis)
            print(
                f"[NLI] 화장품가설{i+1} | entail={entail:.2f} contra={contra:.2f} | \"{preview}\""
            )
            if entail > best_cosmetic_entail:
                best_cosmetic_entail = entail

        # 비화장품 가설 최대 entailment
        best_non_cosmetic_entail = 0.0
        for i, hypothesis in enumerate(NON_COSMETIC_HYPOTHESES):
            entail, contra = _get_nli_probs(text, hypothesis)
            print(
                f"[NLI] 비화장품가설{i+1} | entail={entail:.2f} contra={contra:.2f} | \"{preview}\""
            )
            if entail > best_non_cosmetic_entail:
                best_non_cosmetic_entail = entail

        # 화장품 entail이 비화장품 entail보다 높고 최소 0.15 이상이면 화장품
        is_cosmetic = best_cosmetic_entail > best_non_cosmetic_entail and best_cosmetic_entail >= 0.15
        print(
            f"[NLI] 도메인판별 최종 | 화장품={best_cosmetic_entail:.2f} 비화장품={best_non_cosmetic_entail:.2f} | "
            f"{'✔ 화장품 → 분석 진행' if is_cosmetic else '✘ 화장품 아님 → 분석 중단'}"
        )
        return is_cosmetic
    except Exception as e:
        print(f"[NLI] 도메인 판별 오류: {e}")
        return True


def _get_nli_probs(premise: str, hypothesis: str) -> tuple[float, float]:
    """(entailment 확률, contradiction 확률) 반환"""
    inputs = _nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        probs = torch.softmax(_nli_model(**inputs).logits, dim=-1)[0]
    return probs[ENTAILMENT_IDX].item(), probs[CONTRADICTION_IDX].item()


def _select_hypotheses(
    rule_patterns: list[str],
) -> list[tuple[str, SuspicionLevel]]:
    """규칙 엔진 패턴 태그 기준으로 관련 가설만 선택"""
    selected = []
    for pattern in rule_patterns:
        selected.extend(VIOLATION_HYPOTHESES.get(pattern, []))
    return selected


def verify_with_nli(
    sentence: str,
    kobert_result: SentenceResult,
    rule_patterns: list[str],
) -> SentenceResult:
    """
    NLI로 KoBERT 결과를 검증·보정.

    Args:
        sentence:       분석 대상 문장
        kobert_result:  KoBERT 분석 결과
        rule_patterns:  규칙 엔진이 감지한 패턴 태그 목록
    """
    if not _load_nli_model():
        return kobert_result

    hypotheses = _select_hypotheses(rule_patterns)
    if not hypotheses:
        return kobert_result

    try:
        best_entailment   = 0.0
        best_contradiction = 0.0
        best_level        = None

        for hypothesis, level in hypotheses:
            entail, contra = _get_nli_probs(sentence, hypothesis)
            if entail > best_entailment:
                best_entailment = entail
                best_level      = level
            if contra > best_contradiction:
                best_contradiction = contra

        current_level = kobert_result.suspicion_level
        current_score = kobert_result.score

        # ── NLI 결과 반영 ────────────────────────────────────
        if best_entailment >= NLI_CONFIRM_THRESHOLD:
            # 위반 확인 → 점수 상향, 필요 시 레벨 업그레이드
            new_score = round(current_score * 0.5 + best_entailment * 0.5, 3)
            if best_level and _LEVEL_ORDER[best_level] > _LEVEL_ORDER[current_level]:
                new_level = best_level
                print(
                    f"[NLI] 레벨 상향: {current_level.name} → {new_level.name} "
                    f"(entail={best_entailment:.2f})"
                )
            else:
                new_level = current_level

        elif best_contradiction >= NLI_DENY_THRESHOLD:
            # 위반 부정 → 점수 하향
            new_score = round(current_score * 0.6, 3)
            if (
                current_level == SuspicionLevel.SUSPICIOUS
                and new_score < 0.45
            ):
                new_level = SuspicionLevel.CAUTION
                print(
                    f"[NLI] 레벨 하향: SUSPICIOUS → CAUTION "
                    f"(contra={best_contradiction:.2f})"
                )
            else:
                new_level = current_level

        else:
            # 중립 → KoBERT 결과 유지
            new_score = current_score
            new_level = current_level

        return SentenceResult(
            sentence=sentence,
            suspicion_level=new_level,
            matched_keywords=kobert_result.matched_keywords,
            matched_patterns=kobert_result.matched_patterns,
            reason=kobert_result.reason,
            score=new_score,
        )

    except Exception as e:
        print(f"[NLI] 검증 오류: {e}")
        return kobert_result