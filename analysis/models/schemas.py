from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class InputType(str, Enum):
    TEXT = "text"
    URL = "url"
    IMAGE = "image"


class SuspicionLevel(str, Enum):
    NORMAL = "정상"
    CAUTION = "주의"
    SUSPICIOUS = "의심"


class AnalyzeRequest(BaseModel):
    input_type: InputType
    content: str  # 텍스트 or URL (이미지는 별도 엔드포인트)


class SentenceResult(BaseModel):
    sentence: str
    suspicion_level: SuspicionLevel
    matched_keywords: List[str]
    reason: str


class AnalyzeResponse(BaseModel):
    original_text: str
    overall_suspicion_level: SuspicionLevel
    overall_score: float  # 0.0 ~ 1.0
    sentence_results: List[SentenceResult]
    summary: str
