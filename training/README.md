# KoBERT 광고 분류기 학습 가이드

이 프로젝트는 KoBERT(`skt/kobert-base-v1`)를 기본 모델로 사용해 광고 의심/일반/기타(3클래스) 문장 분류기를 학습합니다.

## 전체 흐름
1. 원본 시트 정리 → `clean_*.csv` 생성  
2. `python merge_clean_sheets.py`로 통합 → `data/merged/merged_clean_YYYYMMDD_HHMMSS.csv`  
3. `python training/train.py` 실행 → 학습 결과가 `models/kobert_ad_classifier/`에 저장  
4. FastAPI 서버가 해당 모델을 로드해 추론

## 데이터 요약
- 입력 파일: `data/merged/merged_clean*.csv` 중 최신 1개 자동 사용  
- 필요한 컬럼  
  - `광고여부선택`: 값은 `광고`, `의심`, `해당없음`  
  - `광고 문장`: 분류할 텍스트  
- 레이블 매핑  
  - `광고`→0, `의심`→1, `해당없음`→2

## 학습 실행
```bash
# 1) 데이터 병합 (필요시)
python merge_clean_sheets.py

# 2) KoBERT 학습
python training/train.py
```
- 체크포인트는 `models/checkpoints_kobert/epoch_*`에 저장/재개됨.
- 최종 모델과 토크나이저는 `models/kobert_ad_classifier/`에 저장됨.

## 의존성
- `requirements.txt`에 KoBERT HF 포트가 포함됨:  
  `git+https://github.com/SKTBrain/KoBERT.git#subdirectory=kobert_hf`
- PyTorch는 GPU 환경에 맞춰 별도 설치 필요 (예시):  
  - CUDA 12.4: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`  
  - CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`  
  - CUDA 11.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`  
  - CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## 기본 설정 확인
- `training/train.py`  
  - `BASE_MODEL = "skt/kobert-base-v1"` (KoBERT 고정)  
  - 저장 경로: `MODEL_SAVE_PATH = "models/kobert_ad_classifier"`  
  - 체크포인트 경로: `CKPT_DIR = "models/checkpoints_kobert"`

## 자주 하는 질문
- **기존 Electra 모델이 저장되던 이유?** 과거 버전의 스크립트/README에서 기본값이 Electra였기 때문. 현재는 KoBERT로 고정.  
- **패키지 설치 오류**: KoBERT HF 포트가 설치되지 않으면 `AutoTokenizer` 로드 시 에러 발생. `pip install ...kobert_hf` 확인.  
- **GPU 없이 가능?** 가능하지만 느립니다. `MAX_LEN`과 `BATCH_SIZE`를 줄이면 메모리 부족을 완화할 수 있습니다.
