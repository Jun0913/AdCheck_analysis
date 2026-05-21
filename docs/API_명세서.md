# 광고체크 API 명세서
**FastAPI 분석 서버(Python) + Spring Boot 백엔드(Java) + React 프론트엔드 연동용**

---

## 기본 정보

| 항목 | 값 |
|---|---|
| Spring Boot 서버 | `http://localhost:8080` |
| FastAPI 분석 서버 | `http://localhost:8000` |
| 데이터 형식 | JSON (UTF-8) |
| 인증 방식 | 없음 (내부 통신) |

---

## 의심도 레벨

| 값 | 설명 |
|---|---|
| `"정상"` | 허위·과장 위험 표현이 확인되지 않음 |
| `"주의"` | 과장 가능성이 있는 표현이 포함됨 |
| `"의심"` | 허위·과장 광고로 오인될 수 있는 표현이 포함됨 |

---

## [프론트엔드→Spring Boot] API 목록

---

### 1. 서버 상태 확인

```
GET /health
```

**응답**
```json
{
  "status": "ok",
  "service": "광고체크 분석 서버",
  "use_kobert": true,
  "kobert_ready": true,
  "kobert_available": true,
  "nli_ready": false,
  "nli_available": false,
  "warmup_in_progress": false,
  "ready": true,
  "startup_error": null
}
```

---

### 1-1. 분석 서버 준비 상태 확인

```
GET /ready
```

- 모델 preload가 완료되면 `200 OK`
- 아직 기동 중이거나 preload 실패면 `503 Service Unavailable`
- 배포 환경의 readiness probe는 `/health`가 아니라 `/ready`를 사용 권장

**준비 완료 응답**
```json
{
  "status": "ok",
  "service": "광고체크 분석 서버",
  "use_kobert": true,
  "kobert_ready": true,
  "kobert_available": true,
  "nli_ready": false,
  "nli_available": false,
  "warmup_in_progress": false,
  "ready": true,
  "startup_error": null
}
```

**준비 전 응답 예시**
```json
{
  "status": "not_ready",
  "service": "광고체크 분석 서버",
  "use_kobert": true,
  "kobert_ready": false,
  "kobert_available": true,
  "nli_ready": false,
  "nli_available": false,
  "warmup_in_progress": true,
  "ready": false,
  "startup_error": null
}
```

---

### 2. 텍스트 분석

```
POST /analyze/text
Content-Type: application/json
```

**요청**
```json
{
  "content": "아토피 피부염 증상 치료에 도움이 되는 클렌저"
}
```

**응답**
```json
{
  "original_text": "아토피 피부염 증상 치료에 도움이 되는 클렌저",
  "overall_suspicion_level": "의심",
  "overall_score": 0.7,
  "sentence_results": [
    {
      "sentence": "아토피 피부염 증상 치료에 도움이 되는 클렌저",
      "suspicion_level": "의심",
      "matched_keywords": ["아토피", "피부염", "치료에 도움", "치료"],
      "matched_patterns": ["의약품오인"],
      "reason": "의약품오인 관련 표현이 감지되어 정책상 최소 의심 단계를 유지합니다.",
      "score": 0.7
    }
  ],
  "summary": "총 1개 문장 중 1개에서 허위·과장 가능성이 높은 표현이 감지되었습니다. 주요 문구: \"아토피 피부염 증상 치료에 도움이 되는 클렌저\" - 의약품오인 관련 표현이 감지되어 정책상 최소 의심 단계를 유지합니다. 광고 내용을 비판적으로 검토하시기 바랍니다."
}
```

---

### 3. URL 분석

```
POST /analyze/url
Content-Type: application/json
```

**요청**
```json
{
  "content": "https://example.com/product"
}
```

**응답**  
텍스트 분석과 동일한 형식으로 반환됩니다.

---

### 4. 이미지 분석

```
POST /analyze/image
Content-Type: multipart/form-data
```

**요청 Form-data**
| 키 | 타입 | 설명 |
|---|---|---|
| `file` | File | 이미지 파일 (jpg, png 등) |

**응답**  
텍스트 분석과 동일한 형식으로 반환됩니다.

---

### 5. 분석 이력 목록

```
GET /history?page=0&size=10
```

**응답**  
페이지네이션된 분석 결과 목록을 반환합니다.

---

### 6. 분석 이력 단건 조회

```
GET /history/{id}
```

---

## 에러 응답 형식

모든 에러는 아래 형식으로 반환됩니다.

```json
{
  "error": "에러 메시지"
}
```

또는 FastAPI 검증/상세 에러는 아래처럼 반환될 수 있습니다.

```json
{
  "detail": "에러 메시지"
}
```

| HTTP 상태코드 | 상황 |
|---|---|
| `400` | 잘못된 요청, URL 추출 실패, 이미지 파일 형식 오류 |
| `422` | 이미지 OCR 실패, 텍스트 인식 실패 |
| `500` | Spring Boot 내부 오류 또는 분석 서버 통신 오류 |

---

## [Spring Boot ↔ FastAPI] 내부 통신

| 엔드포인트 | 용도 |
|---|---|
| `GET /health` | 분석 서버 상태 확인 |
| `GET /ready` | 모델 preload 완료 여부 확인 |
| `POST /analyze/text` | 텍스트 분석 요청 |
| `POST /analyze/url` | URL 분석 요청용 호환 엔드포인트 |
| `POST /analyze/image` | 이미지 분석 요청 |
| `POST /analyze` | 구버전 클라이언트 호환용 기본 분석 엔드포인트 |

### FastAPI 요청 형식

#### `POST /analyze/text`
```json
{
  "input_type": "text",
  "content": "분석할 광고 문구"
}
```

`input_type`는 기본값이 `"text"`이므로, 아래처럼 보내도 동작합니다.

```json
{
  "content": "분석할 광고 문구"
}
```

#### `POST /analyze/text`에서 URL 분석
```json
{
  "input_type": "url",
  "content": "https://example.com/product"
}
```

#### `POST /analyze/url`
```json
{
  "content": "https://example.com/product"
}
```

---

## 전체 통신 흐름

```
React (3000 or 5173)
  ↓  POST /analyze/text
Spring Boot (8080)
  1. 요청 수신 및 인증/권한 처리
  2. FastAPI로 분석 요청 전달
  ↓  POST /analyze/text 또는 /analyze/url 또는 /analyze/image
FastAPI (8000)
  1. 텍스트 추출 (직접 입력 / URL 본문 추출 / 이미지 OCR)
  2. 문장 분리 (kss 우선, 없으면 정규식 폴백)
  3. 규칙 기반 1차 판정
  4. 선택적 KoBERT 문맥 판정
  5. 조건부 NLI 보정 (설정 시)
  6. JSON 결과 반환
  ↓
Spring Boot (8080)
  3. 결과 저장
  4. React로 응답 반환
  ↓
React 결과 화면 표시
```
