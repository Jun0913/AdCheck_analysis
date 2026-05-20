import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from analysis.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    InputType,
    SentenceResult,
    SuspicionLevel,
)
from analysis.services.ad_domain_filter import is_noise, predict_ad, predict_cosmetic
from analysis.services.ai_analyzer import analyze_with_kobert, generate_summary
from analysis.services.extractor import extract_from_image, extract_from_url, split_sentences
from analysis.services.rule_domain import has_image_cosmetic_context
from analysis.services.rule_engine import (
    analyze_sentence,
    calculate_overall_score,
    should_run_kobert,
)


router = APIRouter(prefix="/analyze", tags=["analyze"])
logger = logging.getLogger(__name__)

USE_KOBERT = os.getenv("USE_KOBERT", "true").lower() == "true"


class LegacyContentRequest(BaseModel):
    content: str


@router.post("", response_model=AnalyzeResponse)
async def analyze_default(request: AnalyzeRequest):
    """Compatibility endpoint for callers posting directly to /analyze."""
    return await _analyze_by_input_type(request.content, request.input_type)


@router.post("/text", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text input or a URL passed via input_type=url."""
    return await _analyze_by_input_type(request.content, request.input_type)


@router.post("/url", response_model=AnalyzeResponse)
async def analyze_url(request: LegacyContentRequest):
    """Legacy URL endpoint kept for backend/frontend compatibility."""
    return await _analyze_by_input_type(request.content, InputType.URL)


@router.post("/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image input via OCR."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    image_bytes = await file.read()
    try:
        text = extract_from_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지에서 텍스트를 추출하지 못했습니다. {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="이미지에서 텍스트를 인식하지 못했습니다.")

    return await _run_analysis(text, source_type="image")


async def _analyze_by_input_type(content: str, input_type: InputType) -> AnalyzeResponse:
    if input_type == InputType.URL:
        try:
            text = await extract_from_url(content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL에서 텍스트를 가져오지 못했습니다. {str(e)}")
    else:
        text = content

    return await _run_analysis(text, source_type="text")


async def _run_analysis(text: str, *, source_type: str = "text") -> AnalyzeResponse:
    """Shared analysis pipeline."""
    sentences = split_sentences(text)

    if not sentences:
        return AnalyzeResponse(
            original_text=text,
            overall_suspicion_level=SuspicionLevel.NORMAL,
            overall_score=0.0,
            sentence_results=[],
            summary="분석할 문장이 충분하지 않습니다.",
        )

    rule_results: list[SentenceResult] = []
    cosmetic_predictions = [predict_cosmetic(s) for s in sentences]
    image_has_cosmetic_context = (
        source_type == "image"
        and has_image_cosmetic_context(sentences, cosmetic_predictions)
    )

    for idx, sentence in enumerate(sentences):
        if is_noise(sentence):
            logger.debug(
                "filter_result",
                extra={
                    "sentence_preview": sentence[:80],
                    "is_noise": True,
                    "is_ad": None,
                    "ad_score": None,
                    "force_cosmetic": None,
                    "cosmetic_score": None,
                },
            )
            rule_results.append(
                SentenceResult(
                    sentence=sentence,
                    suspicion_level=SuspicionLevel.NORMAL,
                    matched_keywords=[],
                    matched_patterns=[],
                    reason="의미 없는/너무 짧은 문장으로 판단하여 건너뜀.",
                    score=0.0,
                )
            )
            continue

        cosmetic_pred = cosmetic_predictions[idx]
        force_cosmetic = image_has_cosmetic_context or (
            False if cosmetic_pred is None else cosmetic_pred[0]
        )
        cosmetic_score = None if cosmetic_pred is None else round(cosmetic_pred[1], 3)

        ad_pred = predict_ad(sentence)
        is_ad = True if ad_pred is None else ad_pred[0]
        ad_score = None if ad_pred is None else round(ad_pred[1], 3)

        logger.debug(
            "filter_result",
            extra={
                "sentence_preview": sentence[:80],
                "is_noise": False,
                "is_ad": is_ad,
                "ad_score": ad_score,
                "force_cosmetic": force_cosmetic,
                "cosmetic_score": cosmetic_score,
            },
        )

        rule_results.append(
            analyze_sentence(
                sentence,
                force_cosmetic=force_cosmetic,
                source_type=source_type,
            )
        )

    if USE_KOBERT:
        final_results: list[SentenceResult] = []
        for rule_result in rule_results:
            if not should_run_kobert(rule_result):
                final_results.append(rule_result)
                continue

            kobert_result = await analyze_with_kobert(rule_result.sentence, rule_result)
            final_results.append(kobert_result)
    else:
        final_results = rule_results

    score, overall_level = calculate_overall_score(final_results)
    summary = generate_summary(overall_level, final_results, score)

    return AnalyzeResponse(
        original_text=text,
        overall_suspicion_level=overall_level,
        overall_score=score,
        sentence_results=final_results,
        summary=summary,
    )
