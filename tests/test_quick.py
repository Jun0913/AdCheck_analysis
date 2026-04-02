"""
텍스트 분석 빠른 테스트
딱 걸렸어! 프로젝트

실행 전 FastAPI 서버가 켜져있어야 합니다.
  uvicorn analysis.main:app --port 8000

사용법:
  python tests/test_quick.py
"""
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "name": "정상 광고 문구",
        "content": "히알루론산 성분으로 피부 보습에 도움을 주는 크림입니다. 산뜻한 사용감으로 매일 사용하기 좋습니다.",
    },
    {
        "name": "의약품 오인 표현",
        "content": "이 크림은 아토피를 치료합니다. 피부과 전문의가 추천하는 제품입니다.",
    },
    {
        "name": "절대적·과장 표현",
        "content": "단 7일 만에 기미 완전 제거! 부작용이 전혀 없으며 100% 효과를 보장합니다.",
    },
    {
        "name": "혼합 문구 (정상 + 의심)",
        "content": "촉촉한 사용감의 보습 크림입니다. 세포재생으로 피부 나이를 되돌려드립니다. 피부 보습에 도움을 드립니다.",
    },
]


def run_test(name: str, content: str):
    body = json.dumps({"input_type": "text", "content": content}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/analyze/text",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"\n{'─' * 55}")
    print(f"  테스트: {name}")
    print(f"{'─' * 55}")

    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        icon = {"정상": "✅", "주의": "⚠️ ", "의심": "🚨"}.get(resp["overall_suspicion_level"], "?")
        print(f"  {icon} 전체 의심도: {resp['overall_suspicion_level']} ({resp['overall_score']}점)")
        print(f"  요약: {resp['summary']}")
        for s in resp["sentence_results"]:
            si = {"정상": "✅", "주의": "⚠️ ", "의심": "🚨"}.get(s["suspicion_level"], "?")
            print(f"    {si} [{s['suspicion_level']}] {s['sentence']}")
            if s["matched_keywords"]:
                print(f"       키워드: {', '.join(s['matched_keywords'])}")
    except urllib.error.HTTPError as e:
        print(f"  오류: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"  오류: {e}")


if __name__ == "__main__":
    print("=" * 55)
    print("  딱 걸렸어! 텍스트 분석 테스트")
    print("=" * 55)
    for case in TEST_CASES:
        run_test(case["name"], case["content"])
    print(f"\n{'=' * 55}")
    print("  테스트 완료")
    print("=" * 55)
