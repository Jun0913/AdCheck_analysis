# 딱 걸렸어! — Python 분석 서버 프로젝트 세팅 가이드

새 컴퓨터에서 프로젝트를 처음 시작할 때 이 순서대로 따라하세요.

---

## 개발 환경

| 항목 | 버전 |
|---|---|
| **Python** | **3.11.9** |
| **IDE** | PyCharm (버전 무관) |
| **가상환경** | venv (`.venv/`) |
| **패키지 관리** | pip |

> ⚠️ Python은 반드시 **3.11.x** 버전을 사용하세요.
> 3.12 이상은 일부 패키지(kss, torch 등) 호환성 문제가 생길 수 있어요.

---

## 1. Python 3.11.9 설치

https://www.python.org/downloads/release/python-3119/

- Windows: `Windows installer (64-bit)` 다운로드
- 설치 시 **"Add Python to PATH"** 체크 필수

설치 확인:
```bash
python --version
# Python 3.11.9
```

---

## 2. 프로젝트 클론

```bash
git clone {저장소 URL}
cd analysis
```

---

## 3. PyCharm에서 프로젝트 열기

1. PyCharm 실행 → `Open` → `analysis` 폴더 선택
2. 우측 하단 인터프리터 설정 → `Add New Interpreter` → `Add Local Interpreter`
3. `Virtualenv Environment` → `New` 선택
4. Base interpreter: **Python 3.11.9** 선택
5. Location: 프로젝트 루트의 `.venv` 폴더로 설정

---

## 4. 가상환경 생성 (터미널에서 직접 할 경우)

```bash
# 프로젝트 루트에서 실행
python -m venv .venv

# 활성화 (Windows)
.venv\Scripts\activate

# 활성화 (macOS / Linux)
source .venv/bin/activate
```

---

## 5. PyTorch 설치 (GPU / CPU 선택)

> `requirements.txt`에 torch가 빠져있어요. 환경에 맞게 따로 설치해야 해요.

### GPU 사용 (CUDA) — 학습 시 필수

```bash
# CUDA 버전 확인
nvidia-smi
```

| CUDA 버전 | 설치 명령어 |
|---|---|
| 12.4 | `pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu118` |

### CPU 전용 (서버 실행만 할 경우)

```bash
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 6. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 7. Tesseract OCR 설치 (이미지 분석 기능용)

### Windows

1. https://github.com/UB-Mannheim/tesseract/wiki 에서 설치 파일 다운로드
2. 설치 시 `Additional language data` → **Korean** 선택
3. 설치 경로 확인 (보통 `C:\Program Files\Tesseract-OCR\`)
4. 시스템 환경변수 PATH에 추가

설치 확인:
```bash
tesseract --version
```

### macOS
```bash
brew install tesseract tesseract-lang
```

### Linux (Ubuntu)
```bash
sudo apt install tesseract-ocr tesseract-ocr-kor
```

---

## 8. 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
# KoBERT 모델 경로 (학습 완료 후 모델 파일 위치)
KOBERT_MODEL_PATH=models/kobert_ad_classifier_relabel

# KoBERT 사용 여부 (모델 없으면 false로 설정)
USE_KOBERT=true
```

> 기본 모델 파일(`models/kobert_ad_classifier_relabel/`)은 Git에 포함되어 있지 않아요.
> 팀원에게 드라이브 링크로 받아서 해당 경로에 넣어주세요.
> 모델 없이 실행하려면 `.env`에서 `USE_KOBERT=false` 로 설정하면 규칙기반 엔진만 동작해요.

---

## 9. 설치 확인

```bash
# GPU 인식 확인 (GPU 환경만)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 서버 실행 테스트
python test_server.py
```

---

## 10. 서버 실행

```bash
# 직접 실행
uvicorn analysis.main:app --host 0.0.0.0 --port 8000

# 또는 PyCharm Run Configuration 사용 (test_server.xml 설정 포함되어 있음)
```

실행 후 확인:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Readiness check: http://localhost:8000/ready

---

## 핵심 패키지 버전 목록

> 문제 발생 시 아래 버전으로 맞춰주세요.

| 패키지 | 버전 |
|---|---|
| fastapi | 0.115.0 |
| uvicorn | 0.30.6 |
| pydantic | 2.9.2 |
| httpx | 0.27.2 |
| beautifulsoup4 | 4.12.3 |
| pillow | 10.4.0 |
| pytesseract | 0.3.13 |
| python-multipart | 0.0.9 |
| python-dotenv | 1.0.1 |
| kss | 4.5.4 |
| torch | 2.11.0 |
| transformers | 5.4.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.4 |

---

## 룰 엔진 빠른 점검

정제 시트를 이용해 규칙 엔진이 어느 정도 맞게 동작하는지 간단히 확인하려면:

```
python -m analysis.tools.eval_rule_engine --limit 1000
```

`--limit 0`이면 전체 시트를 사용합니다. 모델 재학습 없이 규칙 수정 효과를 빠르게 보는 회귀 체크용 스크립트입니다.

