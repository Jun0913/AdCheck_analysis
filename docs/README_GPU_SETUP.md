# GPU 환경 세팅 가이드

## 1. CUDA 버전 확인

```bash
nvidia-smi
```

출력 결과 오른쪽 위에 CUDA 버전이 표시됩니다.

## 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

## 3. PyTorch GPU 버전 설치 (CUDA 버전에 맞게 선택)

```bash
# CUDA 12.1 (가장 최신)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 4. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

## 5. GPU 인식 확인

```bash
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

`CUDA 사용 가능: True` 가 나오면 성공입니다.

## 6. 학습 실행

```bash
python training/train.py
```

---

## VRAM 부족할 경우

`training/train.py` 상단의 `BATCH_SIZE`를 줄이세요.

```python
BATCH_SIZE = 32   # 기본값 (VRAM 6GB↑)
BATCH_SIZE = 16   # VRAM 4~6GB
BATCH_SIZE = 8    # VRAM 4GB 미만
```

---

## 학습 완료 후

`models/kobert_ad_classifier/` 폴더가 생성됩니다.
이 폴더를 팀원들과 드라이브로 공유하면 다른 컴퓨터에서 재학습 없이 바로 서버를 실행할 수 있습니다.
