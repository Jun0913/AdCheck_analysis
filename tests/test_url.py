"""
URL 분석 테스트
딱 걸렸어! 프로젝트

실행 전 FastAPI 서버가 켜져있어야 합니다.
  uvicorn analysis.main:app --port 8000

사용법:
  python tests/test_url.py                         ← 기본 테스트 케이스 실행
  python tests/test_url.py https://example.com     ← 직접 URL 지정
"""
import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def analyze_url(url: str):
    body = json.dumps({
        "input_type": "url",
        "content": url
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/analyze/text",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print("=" * 60)
    print(f"분석 URL: {url}")
    print("=" * 60)

    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        print(f"전체 의심도 : {resp['overall_suspicion_level']}  ({resp['overall_score']}점)")
        print(f"요약        : {resp['summary']}")
        print(f"추출 텍스트 : {resp['original_text'][:200]}...")
        print()
        print(f"문장별 결과 ({len(resp['sentence_results'])}개, 상위 5개):")
        for i, s in enumerate(resp["sentence_results"][:5], 1):
            icon = {"정상": "✅", "주의": "⚠️ ", "의심": "🚨"}.get(s["suspicion_level"], "?")
            print(f"  {i}. {icon} [{s['suspicion_level']}] {s['sentence']}")
            if s["matched_keywords"]:
                print(f"     키워드: {', '.join(s['matched_keywords'])}")
    except urllib.error.HTTPError as e:
        error = json.loads(e.read().decode("utf-8"))
        print(f"결과: {error['detail']}")
    except Exception as e:
        print(f"오류: {e}")

    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_url(sys.argv[1])
    else:
        # 차단 사이트 테스트 — 친절한 안내 메시지 확인
        print("[차단 사이트 테스트]")
        analyze_url("https://www.coupang.com/vp/products/123456")
        analyze_url("https://www.youtube.com/watch?v=abc123")

        # 일반 사이트 테스트 — 실제 크롤링 확인
        print("[일반 사이트 테스트]")
        analyze_url("https://blog.naver.com/PostView.naver?blogId=cosmetic_ad&logNo=12345")
