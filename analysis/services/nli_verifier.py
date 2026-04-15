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

import json
import os
from pathlib import Path
from analysis.services.model_runtime import prepare_model_runtime

prepare_model_runtime()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analysis.models.schemas import SentenceResult, SuspicionLevel

DEFAULT_HYPOTHESES_PATH = Path(__file__).resolve().parent.parent / "config" / "nli_hypotheses.json"
NLI_HYPOTHESES_PATH = Path(os.getenv("NLI_HYPOTHESES_PATH", str(DEFAULT_HYPOTHESES_PATH)))
CACHE_MODEL_DIR = (
    Path(".cache")
    / "huggingface"
    / "transformers"
    / "models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTAILMENT_IDX    = 0
CONTRADICTION_IDX = 2

NLI_CONFIRM_THRESHOLD = 0.60  # entailment >= 이 값이면 위반 가설 확인
NLI_DENY_THRESHOLD    = 0.50  # contradiction >= 이 값이면 위반 부정

_nli_tokenizer = None
_nli_model     = None
_hypotheses_config = None


_LEVEL_NAME_MAP = {
    "NORMAL": SuspicionLevel.NORMAL,
    "CAUTION": SuspicionLevel.CAUTION,
    "SUSPICIOUS": SuspicionLevel.SUSPICIOUS,
}


def _resolve_default_nli_model_path() -> str:
    snapshots_dir = CACHE_MODEL_DIR / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


NLI_MODEL_NAME = os.getenv("NLI_MODEL_PATH", _resolve_default_nli_model_path())


def _load_nli_model() -> bool:
    global _nli_tokenizer, _nli_model
    if _nli_model is not None:
        return True
    prepare_model_runtime()
    try:
        local_only = os.path.exists(NLI_MODEL_NAME)
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, local_files_only=local_only)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME,
            local_files_only=local_only,
        ).to(DEVICE)
        _nli_model.eval()
        print(f"[NLI] 모델 로드 완료: {NLI_MODEL_NAME}")
        return True
    except Exception as e:
        print(f"[NLI] 모델 로드 실패: {e}")
        return False


def _load_hypotheses_config() -> dict[str, list[dict]]:
    global _hypotheses_config
    if _hypotheses_config is not None:
        return _hypotheses_config

    if not NLI_HYPOTHESES_PATH.exists():
        print(f"[NLI] 가설 설정 파일 없음: {NLI_HYPOTHESES_PATH}")
        _hypotheses_config = {}
        return _hypotheses_config

    try:
        with NLI_HYPOTHESES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[NLI] 가설 설정 파일 로드 실패: {e}")
        _hypotheses_config = {}
        return _hypotheses_config

    normalized: dict[str, list[dict]] = {}
    for pattern, items in raw.items():
        valid_items = []
        for item in items:
            level_name = str(item.get("level", "CAUTION")).upper()
            hypothesis = str(item.get("hypothesis", "")).strip()
            if not hypothesis or level_name not in _LEVEL_NAME_MAP:
                continue
            valid_items.append(
                {
                    "hypothesis": hypothesis,
                    "level": _LEVEL_NAME_MAP[level_name],
                    "weight": float(item.get("weight", 1.0)),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        normalized[pattern] = valid_items

    _hypotheses_config = normalized
    return _hypotheses_config

_LEVEL_ORDER = {
    SuspicionLevel.NORMAL:     0,
    SuspicionLevel.CAUTION:    1,
    SuspicionLevel.SUSPICIOUS: 2,
}


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
) -> list[dict]:
    """규칙 엔진 패턴 태그 기준으로 관련 가설만 선택"""
    hypotheses_config = _load_hypotheses_config()
    selected = []
    for pattern in rule_patterns:
        for item in hypotheses_config.get(pattern, []):
            if item["enabled"]:
                selected.append(item)
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

        for item in hypotheses:
            entail, contra = _get_nli_probs(sentence, item["hypothesis"])
            weighted_entail = entail * item["weight"]
            if weighted_entail > best_entailment:
                best_entailment = weighted_entail
                best_level      = item["level"]
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
