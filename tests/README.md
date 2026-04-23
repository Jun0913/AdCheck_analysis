# 테스트 가이드

대부분의 테스트는 FastAPI `TestClient`를 사용하므로 별도 서버 실행 없이 실행할 수 있습니다.

## 전체 테스트

```bash
pytest -q
```

## 주요 테스트 파일

- `tests/test_api_analyze.py`: API 엔드포인트, URL/이미지 입력 처리
- `tests/test_pipeline_regression.py`: 룰 엔진 + KoBERT 파이프라인 회귀 테스트
- `tests/test_ocr_postprocess.py`: OCR 후처리 및 문장 병합 테스트

## 수동 서버 테스트

실제 서버를 띄워 수동으로 확인하려면 아래 명령을 사용합니다.

```bash
uvicorn analysis.main:app --port 8000
python test_server.py
```
