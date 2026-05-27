from enum import Enum

from pydantic import BaseModel


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class SuspicionLevel(str, Enum):
    NORMAL = "정상"
    CAUTION = "주의"
    SUSPICIOUS = "의심"


class AnalyzeRequest(BaseModel):
    # Default to text so older callers can send only {"content": "..."}.
    input_type: InputType = InputType.TEXT
    content: str


class SentenceResult(BaseModel):
    sentence: str
    suspicion_level: SuspicionLevel
    matched_keywords: list[str]
    matched_patterns: list[str] = []
    reason: str
    score: float = 0.0


class AnalyzeResponse(BaseModel):
    original_text: str
    overall_suspicion_level: SuspicionLevel
    overall_score: float
    sentence_results: list[SentenceResult]
    summary: str
