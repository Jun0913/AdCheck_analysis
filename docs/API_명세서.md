# 딱 걸렸어! — API 명세서
**FastAPI 분석 서버 (Python) ↔ Spring Boot 백엔드 (Java) ↔ React 프론트엔드 연동용**

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
| `"정상"` | 허위·과장 표현 없음 |
| `"주의"` | 과장 가능성 있는 표현 포함 |
| `"의심"` | 허위·과장 광고 의심 표현 포함 |

---

## [프론트 → Spring Boot] API 목록

---

### 1. 서버 상태 확인

```
GET /health
```

**응답**
```json
{
  "status": "ok",
  "service": "딱 걸렸어! Spring Boot 서버",
  "python_server": "ok"
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
  "content": "분석할 광고 문구 텍스트"
}
```

**응답**
```json
{
  "original_text": "입력된 원본 텍스트",
  "overall_suspicion_level": "의심",
  "overall_score": 0.85,
  "sentence_results": [
    {
      "sentence": "이 크림은 아토피를 치료합니다.",
      "suspicion_level": "의심",
      "matched_keywords": ["치료", "아토피"],
      "reason": "의약품으로 오인될 수 있는 표현이 포함되어 있습니다."
    }
  ],
  "summary": "총 2개 문장 중 1개에서 허위·과장 가능성이 높은 표현이 감지되었습니다."
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

**응답** — 텍스트 분석과 동일한 형식

---

### 4. 이미지 분석

```
POST /analyze/image
Content-Type: multipart/form-data
```

**요청 Form-data**
| 키 | 타입 | 설명 |
|---|---|---|
| `file` | File | 이미지 파일 (jpg, png 등, 최대 10MB) |

**응답** — 텍스트 분석과 동일한 형식

---

### 5. 분석 이력 목록

```
GET /history?page=0&size=10
```

**응답** — 페이징된 분석 결과 목록 (최신순)

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

| HTTP 상태코드 | 상황 |
|---|---|
| `400` | 요청값 오류 (content 없음, 파일 없음 등) |
| `500` | 분석 서버 통신 오류, 내부 오류 |

---

## [Spring Boot → FastAPI] 내부 통신

| 엔드포인트 | 용도 |
|---|---|
| `GET /health` | 분석 서버 상태 확인 |
| `POST /analyze/text` | 텍스트/URL 분석 요청 |
| `POST /analyze/image` | 이미지 분석 요청 |

---

## 전체 통신 흐름

```
React (3000 or 5173)
  │
  │  POST /analyze/text
  ▼
Spring Boot (8080)
  1. 요청 수신 및 유효성 검사
  2. FastAPI로 분석 요청 전달
  │
  │  POST /analyze/text
  ▼
FastAPI (8000)
  1. 텍스트 추출 (URL 크롤링 / OCR / 직접 입력)
  2. 문장 분리 (kss)
  3. 1차 규칙기반 엔진
  4. 2차 KoBERT 모델
  5. JSON 결과 반환
  │
  ▼
Spring Boot (8080)
  3. 결과 MySQL 저장
  4. React에 응답 반환
  │
  ▼
React — 결과 화면 표시
```
