from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from analysis.models.schemas import AnalyzeRequest, AnalyzeResponse, InputType, SuspicionLevel
from analysis.services.extractor import extract_from_url, extract_from_image, split_sentences
from analysis.services.rule_engine import analyze_sentence, calculate_overall_score
from analysis.services.ai_analyzer import analyze_with_kobert, generate_summary
from analysis.services.nli_verifier import is_cosmetic_ad
import os
import base64

router = APIRouter(prefix="/analyze", tags=["analyze"])

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

    return await _run_analysis(text)


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

    return await _run_analysis(text)


async def _run_analysis(text: str) -> AnalyzeResponse:
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

    # 0차: 화장품·뷰티 광고 도메인 판별 (전체 텍스트에 1번만 실행)
    if not is_cosmetic_ad(text[:512]):  # 앞 512자만 사용해 속도 최적화
        from analysis.models.schemas import SentenceResult
        not_cosmetic_results = [
            SentenceResult(
                sentence=s,
                suspicion_level=SuspicionLevel.NORMAL,
                matched_keywords=[],
                matched_patterns=[],
                reason="화장품·뷰티 광고와 관련 없는 문구로 판단되어 분석 대상에서 제외되었습니다.",
                score=0.0,
            )
            for s in sentences
        ]
        return AnalyzeResponse(
            original_text=text,
            overall_suspicion_level=SuspicionLevel.NORMAL,
            overall_score=0.0,
            sentence_results=not_cosmetic_results,
            summary="화장품·뷰티 광고가 아닌 것으로 판단됩니다. 본 서비스는 화장품 광고의 허위·과장 표현을 분석합니다.",
        )

    # 1차 규칙기반 분석
    rule_results = [analyze_sentence(s) for s in sentences]

    # 2차 KoBERT 분석 (설정에 따라)
    if USE_KOBERT:
        final_results = []
        for rule_result in rule_results:
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
