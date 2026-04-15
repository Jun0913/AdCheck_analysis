# 📌 딱 걸렸어! — 허위·과장 광고 의심도 분석 서비스

## 1. 프로젝트 개요
본 프로젝트는 광고 문구, URL, 이미지 등을 입력받아  
AI가 광고 내용을 분석하고 허위·과장 가능성을 탐지하는 웹 서비스다.

- 0차: 전단 필터 (잡음 컷 + 광고/화장품 도메인 선별)
- 1차: 규칙 기반 엔진 (키워드 + 도메인 필터)
- 2차: KoBERT 모델 (문맥 분석)
- 3차: NLI 검증 레이어 (클레임 의미 검증, 조건부 호출)
- 출력: 주의 / 의심 / 정상 + 이유 설명 + 연속 의심도 점수

---

## 2. 핵심 목표
- 광고 문구 내 허위·과장 표현 탐지
- 문장 단위 분석 및 설명 제공
- 광고 전체 의심도 요약
- 소비자의 비판적 판단 지원

---

## 3. 시스템 구조

Frontend (React)
        ↓
Spring Boot Backend
        ↓
Python Analysis Server (FastAPI)
        ↓
Pre-filter → Rule Engine → KoBERT → NLI Verifier

---

## 4. 동작 흐름

1. 사용자 입력 (텍스트 / URL / 이미지)
2. Backend → Analysis Server 요청
3. 텍스트 추출 (EasyOCR / 크롤링)
4. 문장 분리
5. [0차] 전단 필터 — 잡음 컷 + 광고/화장품 도메인 선별
6. [1차] 규칙 기반 엔진 — 키워드·도메인 필터
7. [2차] KoBERT 모델 — 문맥 분석
8. [3차] NLI 검증 — 애매한 케이스 클레임 의미 검증 (조건부)
9. 의심도 점수 집계 및 설명 생성
10. 결과 반환

---

## 5. 데이터 파이프라인

원본 데이터 수집
        ↓
원본 시트 (RAW)
        ↓
정제 시트 (CLEAN)
        ↓
병합 시트 생성 (`merge_clean_sheets.py`)
        ↓
전단 필터 학습 (`training/train_ad_domain.py`)
        ↓
KoBERT 학습 (`training/train.py`)
        ↓
Analysis Server 추론

---

## 6. 원본 시트 (Raw Data)

- 원문 수정 금지
- 결측값은 NA
- 문장 1개 = 1행
- 허용/금지 문구 모두 수집

---

## 7. 정제 시트 (Clean Data)

- 기준 라벨: 금지=0, 허용=1, 그 외 NA
- 의심도 라벨: 주의 / 의심 / 정상

---

## 8. AI 분석 구조

### 0차: 전단 필터
잡음 컷 + 광고/화장품 도메인 필터로 분석 대상 문장을 먼저 선별

### 1차: 규칙 기반 엔진
화이트리스트·도메인 필터 → 금지 키워드 검사 → 패턴 태그별 의심도 판정

### 2차: KoBERT 모델
문장 전체 문맥 분석 → 연속 의심도 점수 산출 (0.0 ~ 1.0)

### 3차: NLI 검증 레이어
규칙 엔진 패턴 감지 + KoBERT 점수가 회색지대(0.35~0.65)이거나 두 결과가 엇갈릴 때만 호출
위반 가설 텍스트와 대조하여 클레임 의미를 검증하고 최종 결과 보정

---

## 9. 기술 스택

- Frontend: React
- Backend: Spring Boot
- Analysis: Python (FastAPI), scikit-learn, KoBERT, mDeBERTa (NLI)
- OCR: EasyOCR
- DB: MySQL / PostgreSQL

---

## 10. API 문서

프로젝트 API는 Backend와 Analysis Server가 분담한다.

- Backend: 프론트엔드와 직접 연동되는 메인 API 제공
- Analysis Server: 광고 문구 분석 전용 API 제공
- Analysis Server Swagger UI: `/docs`

---

## 11. 실행 방법

1. Frontend(React) 실행
2. Backend(Spring Boot) 실행
3. Analysis Server(FastAPI) 실행
4. Frontend → Backend → Analysis Server 순서로 연동 확인

세부 환경 설정과 실행 명령은 각 파트의 개별 문서를 참고한다.

---

## 12. 한 줄 요약

광고 문장을 입력하면 전단 필터 + 규칙 기반 + KoBERT + NLI 다단계 AI가 허위·과장 여부와 그 이유까지 설명해주는 시스템
