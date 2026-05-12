from fastapi import APIRouter, UploadFile, File, HTTPException
from analysis.models.schemas import AnalyzeRequest, AnalyzeResponse, InputType, SuspicionLevel, SentenceResult
from analysis.services.extractor import extract_from_url, extract_from_image, split_sentences
from analysis.services.rule_engine import analyze_sentence, calculate_overall_score, should_run_kobert
from analysis.services.ad_domain_filter import predict_ad, predict_cosmetic, is_noise
from analysis.services.ai_analyzer import analyze_with_kobert, generate_summary
from analysis.services.rule_domain import has_image_cosmetic_context
import os
import logging

router = APIRouter(prefix="/analyze", tags=["analyze"])
logger = logging.getLogger(__name__)

USE_KOBERT = os.getenv("USE_KOBERT", "true").lower() == "true"


@router.post("/text", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """텍스트 또는 URL 입력 분석"""
    if request.input_type == InputType.URL:
        try:
            text = await extract_from_url(request.content)
        except ValueError as e:
            # 차단 사이트 또는 크롤링 실패 — 사용자 친화적 메시지
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL에서 텍스트를 가져오지 못했습니다: {str(e)}")
    else:
        text = request.content

    return await _run_analysis(text, source_type="text")


@router.post("/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """이미지 업로드 분석 (OCR)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    image_bytes = await file.read()
    try:
        text = extract_from_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지에서 텍스트를 추출하지 못했습니다: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="이미지에서 텍스트를 인식하지 못했습니다.")

    return await _run_analysis(text, source_type="image")


async def _run_analysis(text: str, *, source_type: str = "text") -> AnalyzeResponse:
    """공통 분석 파이프라인"""
    sentences = split_sentences(text)

    if not sentences:
        return AnalyzeResponse(
            original_text=text,
            overall_suspicion_level=SuspicionLevel.NORMAL,
            overall_score=0.0,
            sentence_results=[],
            summary="분석할 문장이 충분하지 않습니다.",
        )

    # 1차 전단 필터 + 룰엔진
    rule_results = []
    cosmetic_predictions = [predict_cosmetic(s) for s in sentences]
    image_has_cosmetic_context = (
        source_type == "image"
        and has_image_cosmetic_context(sentences, cosmetic_predictions)
    )

    for idx, s in enumerate(sentences):
        # 0차: 잡음 컷
        if is_noise(s):
            logger.debug(
                "filter_result",
                extra={
                    "sentence_preview": s[:80],
                    "is_noise": True,
                    "is_ad": None,
                    "ad_score": None,
                    "force_cosmetic": None,
                    "cosmetic_score": None,
                },
            )
            rule_results.append(
                SentenceResult(
                    sentence=s,
                    suspicion_level=SuspicionLevel.NORMAL,
                    matched_keywords=[],
                    matched_patterns=[],
                    reason="의미 없는/너무 짧은 문장으로 판단하여 건너뜀.",
                    score=0.0,
                )
            )
            continue

        cosmetic_pred = cosmetic_predictions[idx]
        force_cosmetic = image_has_cosmetic_context or (False if cosmetic_pred is None else cosmetic_pred[0])
        cos_score = None if cosmetic_pred is None else round(cosmetic_pred[1], 3)

        # 광고 여부 예측은 참고 신호로만 사용한다.
        # false negative가 많아도 룰 엔진/KoBERT가 한 번 더 볼 수 있게
        # 여기서 바로 탈락시키지 않는다.
        ad_pred = predict_ad(s)
        is_ad = True if ad_pred is None else ad_pred[0]
        ad_score = None if ad_pred is None else round(ad_pred[1], 3)

        logger.debug(
            "filter_result",
            extra={
                "sentence_preview": s[:80],
                "is_noise": False,
                "is_ad": is_ad,
                "ad_score": ad_score,
                "force_cosmetic": force_cosmetic,
                "cosmetic_score": cos_score,
            },
        )

        rule_results.append(
            analyze_sentence(
                s,
                force_cosmetic=force_cosmetic,
                source_type=source_type,
            )
        )

    # 2차 KoBERT 분석 (설정에 따라)
    if USE_KOBERT:
        final_results = []
        for rule_result in rule_results:
            if not should_run_kobert(rule_result):
                final_results.append(rule_result)
                continue

            kobert_result = await analyze_with_kobert(rule_result.sentence, rule_result)
            final_results.append(kobert_result)
    else:
        final_results = rule_results

    # 전체 점수 계산
    score, overall_level = calculate_overall_score(final_results)

    # 요약 생성 (외부 API 없이 규칙 기반)
    summary = generate_summary(overall_level, final_results, score)

    return AnalyzeResponse(
        original_text=text,
        overall_suspicion_level=overall_level,
        overall_score=score,
        sentence_results=final_results,
        summary=summary,
    )
