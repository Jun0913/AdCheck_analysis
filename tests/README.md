# 테스트 가이드

FastAPI 분석 서버가 실행 중인 상태에서 아래 테스트를 실행하세요.

## 서버 먼저 실행

```bash
# PyCharm에서 test_server.py 실행
# 또는 터미널에서
uvicorn analysis.main:app --port 8000
```

## 테스트 실행

```bash
# 텍스트 분석 테스트
python tests/test_quick.py

# URL 분석 테스트
python tests/test_url.py
```

## test_server.py (루트)

서버 자동 실행 + 전체 시나리오 테스트를 한 번에 실행합니다.

```bash
python test_server.py
```
