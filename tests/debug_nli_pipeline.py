import asyncio

from analysis.services.ad_domain_filter import predict_ad, predict_cosmetic, is_noise
from analysis.services.ai_analyzer import analyze_with_kobert
from analysis.services.rule_engine import analyze_sentence


SAMPLES = [
    "이 크림은 아토피를 치료합니다.",
    "단 7일 만에 기미 완전 제거, 100% 효과 보장.",
    "피부 보습에 도움을 주는 크림입니다.",
    "피부과 테스트 완료, 촉촉한 사용감의 로션입니다.",
    "의사가 추천하는 세럼으로 주름이 반드시 사라집니다.",
    "줄기세포 기술로 피부를 재생시키는 앰플입니다.",
]


async def main():
    for text in SAMPLES:
        ad = predict_ad(text)
        cosmetic = predict_cosmetic(text)
        rule = analyze_sentence(text, force_cosmetic=False if cosmetic is None else cosmetic[0])
        final = await analyze_with_kobert(text, rule)
        print("=" * 80)
        print(text)
        print("noise=", is_noise(text), "ad=", ad, "cosmetic=", cosmetic)
        print("rule=", rule.suspicion_level.value, rule.score, rule.matched_patterns, rule.reason)
        print("final=", final.suspicion_level.value, final.score, final.matched_patterns, final.reason)


if __name__ == "__main__":
    asyncio.run(main())
