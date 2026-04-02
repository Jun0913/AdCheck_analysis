"""
FastAPI 서버 자동 실행 + 동작 테스트 스크립트
딱 걸렸어! 프로젝트

사용법:
  python test_server.py       ← 서버 자동 실행 + 테스트 한 번에
"""

import json
import time
import subprocess
import sys
import os
import urllib.request
import urllib.error
import signal
import atexit

BASE_URL = "http://localhost:8000"
_server_process = None


def start_server():
    """uvicorn 서버를 백그라운드로 자동 실행"""
    global _server_process

    print("🚀 FastAPI 서버 시작 중...")

    _server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "analysis.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # 서버 뜰 때까지 최대 10초 대기
    for i in range(10):
        time.sleep(1)
        try:
            urllib.request.urlopen(BASE_URL + "/health", timeout=2)
            print(f"✅ 서버 준비 완료 (약 {i+1}초 소요)\n")
            return True
        except:
            print(f"  대기 중... ({i+1}/10)")

    print("❌ 서버 시작 실패")
    return False


def stop_server():
    """테스트 종료 시 서버 자동 종료"""
    global _server_process
    if _server_process:
        _server_process.terminate()
        print("\n🛑 서버 종료")


# 스크립트 종료 시 서버도 자동 종료
atexit.register(stop_server)




def request(method: str, path: str, body: dict = None) -> dict:
    url = BASE_URL + path
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}


def print_result(title: str, result: dict):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")

    if "error" in result:
        print(f"  ❌ 오류: {result}")
        return

    # 분석 결과 출력
    if "overall_suspicion_level" in result:
        print(f"  전체 의심도 : {result['overall_suspicion_level']}  ({result['overall_score']}점)")
        print(f"  요약        : {result['summary']}")
        print(f"\n  문장별 결과:")
        for i, s in enumerate(result.get("sentence_results", []), 1):
            icon = {"정상": "✅", "주의": "⚠️ ", "의심": "🚨"}.get(s["suspicion_level"], "?")
            print(f"    {i}. {icon} [{s['suspicion_level']}] {s['sentence']}")
            print(f"       이유: {s['reason']}")
            if s["matched_keywords"]:
                print(f"       키워드: {', '.join(s['matched_keywords'])}")
    else:
        print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")


def run_tests():
    print("=" * 55)
    print("  딱 걸렸어! FastAPI 서버 테스트")
    print("=" * 55)

    # 서버 자동 실행
    if not start_server():
        return

    # ── 2. 정상 문구 테스트
    print_result(
        "테스트 2: 정상 광고 문구",
        request("POST", "/analyze/text", {
            "input_type": "text",
            "content": "히알루론산 성분으로 피부 보습에 도움을 주는 크림입니다. 산뜻한 사용감으로 매일 사용하기 좋습니다."
        })
    )

    # ── 3. 의심 문구 테스트 (의약품 오인)
    print_result(
        "테스트 3: 의약품 오인 표현",
        request("POST", "/analyze/text", {
            "input_type": "text",
            "content": "이 크림은 아토피를 치료합니다. 피부과 전문의가 추천하는 제품입니다."
        })
    )

    # ── 4. 절대적 표현 테스트
    print_result(
        "테스트 4: 절대적·과장 표현",
        request("POST", "/analyze/text", {
            "input_type": "text",
            "content": "단 7일 만에 기미 완전 제거! 부작용이 전혀 없으며 100% 효과를 보장합니다."
        })
    )

    # ── 5. 혼합 문구 테스트
    print_result(
        "테스트 5: 혼합 문구 (정상 + 의심 혼재)",
        request("POST", "/analyze/text", {
            "input_type": "text",
            "content": "촉촉한 사용감의 보습 크림입니다. 세포재생으로 피부 나이를 되돌려드립니다. 피부 보습에 도움을 드립니다."
        })
    )

    # ── 6. 추천·인증 표현 테스트
    print_result(
        "테스트 6: 추천·인증 표현",
        request("POST", "/analyze/text", {
            "input_type": "text",
            "content": "OO 병원에서 추천하는 화장품. 의사가 직접 처방하는 성분 함유. 임상 시험으로 효과 입증."
        })
    )

    print(f"\n{'=' * 55}")
    print("  테스트 완료!")
    print(f"  Swagger UI: {BASE_URL}/docs")
    print(f"  API 문서:   {BASE_URL}/redoc")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_tests()
