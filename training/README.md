# KoBERT 학습 가이드

## 전체 흐름

```
1. 원본 시트 수집 (raw_*.csv)
       ↓
2. 정제 시트 생성 (clean_*.csv)  ← 프롬프트 문서 참고
       ↓
3. data/clean/ 폴더에 정제 시트 배치
       ↓
4. python merge_clean_sheets.py  (병합 실행 → data/merged/ 에 저장)
       ↓
5. python training/train.py      (학습 실행 → data/merged/ 의 최신 파일 자동 사용)
       ↓
6. models/kobert_ad_classifier/ 에 모델 저장
       ↓
7. FastAPI 서버 실행 시 자동 로드
```

## 데이터 준비

- `data/clean/` 폴더에 `clean_{작업자ID}_{문서번호}.csv` 파일을 넣는다.
- `python merge_clean_sheets.py` 를 실행하면 `data/merged/merged_clean_YYYYMMDD_HHMMSS.csv` 가 생성된다.
- `train.py` 는 `data/merged/` 에서 **가장 최신 병합 파일 하나**를 자동으로 찾아 사용한다.
- 정제 시트 컬럼 중 반드시 필요한 컬럼:
  - `정제 문구` : 학습 입력 텍스트
  - `의심도 라벨` : 정상 / 주의 / 의심

## 라벨 규칙

| 의심도 라벨 | 숫자 라벨 |
|-----------|---------|
| 정상       | 0       |
| 주의       | 1       |
| 의심       | 2       |

## 실행 순서

```bash
# 1. 병합
python merge_clean_sheets.py

# 2. 학습
python training/train.py
```

## 기본 모델

기본값으로 `snunlp/KR-ELECTRA-discriminator` 사용 (HuggingFace 공개 한국어 모델).
순수 KoBERT(`skt/kobert-base-v1`)로 변경하려면 `train.py`의 `BASE_MODEL` 값을 수정한다.
단, `skt/kobert-base-v1`은 별도 토크나이저 설치가 필요하다:
```bash
pip install 'git+https://github.com/SKTBrain/KoBERT.git#subdirectory=kobert_hf'
```
