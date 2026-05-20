import asyncio

import torch

from analysis.models.schemas import SentenceResult, SuspicionLevel
from analysis.services import ai_analyzer


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _DummyModel:
    def __call__(self, **kwargs):
        class Output:
            logits = torch.tensor([[0.1, 0.2, 5.0]])

        return Output()


def test_kobert_non_normal_level_does_not_keep_normal_reason(monkeypatch):
    monkeypatch.setattr(ai_analyzer, "_load_model", lambda: True)
    monkeypatch.setattr(ai_analyzer, "_tokenizer", _DummyTokenizer())
    monkeypatch.setattr(ai_analyzer, "_model", _DummyModel())

    rule_result = SentenceResult(
        sentence="아토피 피부염 증상 치료에 도움이 되는 클렌징폼",
        suspicion_level=SuspicionLevel.NORMAL,
        matched_keywords=["아토피", "피부염", "치료"],
        matched_patterns=[],
        reason="현재 기준에서는 문제 표현이 확인되지 않았습니다.",
        score=0.0,
    )

    final = asyncio.run(ai_analyzer.analyze_with_kobert(rule_result.sentence, rule_result))

    assert final.suspicion_level != SuspicionLevel.NORMAL
    assert final.reason != "현재 기준에서는 문제 표현이 확인되지 않았습니다."
    assert final.matched_patterns


def test_minimum_pattern_level_rewrites_normal_reason():
    result = SentenceResult(
        sentence="아토피 피부염 증상 치료에 도움이 되는 클렌저",
        suspicion_level=SuspicionLevel.NORMAL,
        matched_keywords=["아토피", "피부염", "치료"],
        matched_patterns=["의약품오인"],
        reason="현재 기준에서는 문제 표현이 확인되지 않았습니다.",
        score=0.0,
    )

    adjusted = ai_analyzer._enforce_minimum_pattern_level(result)

    assert adjusted.suspicion_level == SuspicionLevel.SUSPICIOUS
    assert adjusted.reason == "의약품오인 관련 표현이 감지되어 정책상 최소 의심 단계를 유지합니다."
