"""
KoBERT 학습 스크립트 (GPU 최적화)
- 입력: 병합된 정제 시트 CSV (data/merged/merged_clean.csv)
- 출력: models/kobert_ad_classifier/ (학습된 모델)

실행:
    python training/train.py

환경 설정 (처음 한 번만):
    # CUDA 버전 확인
    nvidia-smi

    # PyTorch GPU 버전 설치 (CUDA 버전에 맞게 선택)
    # CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    # CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # 나머지 패키지
    pip install -r requirements.txt

라벨 규칙:
    의심도 라벨 "정상" → 0
    의심도 라벨 "주의" → 1
    의심도 라벨 "의심" → 2
"""

import os
import glob
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────

BASE_MODEL      = "snunlp/KR-ELECTRA-discriminator"
MERGED_PATH     = "data/merged/merged_clean.csv"
MODEL_SAVE_PATH = "models/kobert_ad_classifier"
CKPT_DIR        = "models/checkpoints"

NUM_LABELS  = 3
MAX_LEN     = 128   # GPU: 128 (CPU보다 길게 잡아도 빠름)
BATCH_SIZE  = 32    # GPU: 32 (VRAM 4GB↑ 기준, 부족하면 16으로)
EPOCHS      = 5     # GPU: 5 (CPU보다 여유 있게)
LR          = 2e-5  # 에폭 늘린 만큼 학습률 살짝 낮춤

LABEL_MAP   = {"정상": 0, "주의": 1, "의심": 2}
LABEL_NAMES = ["정상", "주의", "의심"]


# ────────────────────────────────────────────
# 디바이스 설정
# ────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU 사용: {gpu_name} (VRAM: {vram:.1f}GB)")
    else:
        device = torch.device("cpu")
        print("  ⚠️  GPU를 찾을 수 없어 CPU로 실행합니다.")
        print("  PyTorch CUDA 버전이 설치되어 있는지 확인하세요.")
        print("  확인: python -c \"import torch; print(torch.cuda.is_available())\"")
    return device


# ────────────────────────────────────────────
# 데이터 로드 & 전처리
# ────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[오류] 병합 파일 없음: {path}\n"
            "먼저 아래 명령어를 실행하세요:\n"
            "  python merge_clean_sheets.py"
        )
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    print(f"  병합 파일 로드: {path}")
    print(f"  전체 데이터: {len(df)}행")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["의심도 라벨"].isin(LABEL_MAP.keys())].copy()
    df = df[df["정제 문구"].notna() & (df["정제 문구"] != "NA")].copy()
    df["label"] = df["의심도 라벨"].map(LABEL_MAP)
    df = df[["정제 문구", "label"]].rename(columns={"정제 문구": "text"})
    df = df.dropna().reset_index(drop=True)

    print(f"\n  전처리 후 데이터: {len(df)}행")
    print("  라벨 분포:")
    for name, idx in LABEL_MAP.items():
        cnt = (df["label"] == idx).sum()
        print(f"    {name}({idx}): {cnt}건 ({cnt/len(df)*100:.1f}%)")
    return df


# ────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────

class AdDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ────────────────────────────────────────────
# 체크포인트 저장 / 불러오기
# ────────────────────────────────────────────

def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, loss):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch}")
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, os.path.join(ckpt_path, "trainer_state.pt"))
    print(f"  체크포인트 저장: {ckpt_path}")


def find_latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None, 0
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*")))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    epoch = int(os.path.basename(latest).split("_")[1])
    return latest, epoch


# ────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────

def evaluate(model, val_loader, device, epoch):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            pred           = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            preds.extend(pred)
            true_labels.extend(batch["labels"].tolist())

    print(f"\n  [Epoch {epoch} 검증 결과]")
    print(classification_report(
        true_labels, preds,
        target_names=LABEL_NAMES,
        zero_division=0
    ))


# ────────────────────────────────────────────
# 학습 메인
# ────────────────────────────────────────────

def train():
    print("\n" + "=" * 55)
    print("  KoBERT 광고 의심도 분류기 학습 시작 (GPU 최적화)")
    print("=" * 55)

    device = get_device()

    # 데이터 준비
    df = load_data(MERGED_PATH)
    df = preprocess(df)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\n  학습: {len(train_df)}행 / 검증: {len(val_df)}행")

    # 체크포인트 확인
    ckpt_path, start_epoch = find_latest_checkpoint()
    if ckpt_path:
        print(f"\n  이전 체크포인트 발견 (Epoch {start_epoch}): {ckpt_path}")
        print(f"  Epoch {start_epoch + 1}부터 이어서 학습합니다.")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model     = AutoModelForSequenceClassification.from_pretrained(
            ckpt_path, num_labels=NUM_LABELS
        )
    else:
        print(f"\n  모델 로드: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=NUM_LABELS
        )

    model.to(device)

    # GPU면 num_workers 늘려서 데이터 로딩 병렬화
    num_workers = 4 if device.type == "cuda" else 0

    train_dataset = AdDataset(train_df["text"], train_df["label"], tokenizer, MAX_LEN)
    val_dataset   = AdDataset(val_df["text"],   val_df["label"],   tokenizer, MAX_LEN)
    train_loader  = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
    val_loader    = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    )

    # Optimizer & Scheduler
    optimizer    = AdamW(model.parameters(), lr=LR)
    total_steps  = len(train_loader) * (EPOCHS - start_epoch)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    if ckpt_path:
        state = torch.load(
            os.path.join(ckpt_path, "trainer_state.pt"), map_location=device
        )
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

    steps_per_epoch = len(train_loader)
    print(f"\n  배치 크기: {BATCH_SIZE} / 에폭당 스텝: {steps_per_epoch}")
    print(f"  남은 에폭: {EPOCHS - start_epoch}개")
    if device.type == "cuda":
        print(f"  GPU 학습 예상 시간: 에폭당 약 1~3분 (총 {(EPOCHS - start_epoch) * 3}분 이내)")
    print()

    total_start = time.time()

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 10 == 0 or step == steps_per_epoch:
                elapsed  = time.time() - epoch_start
                eta      = elapsed / step * (steps_per_epoch - step)
                avg_loss = total_loss / step
                print(
                    f"\r  Epoch {epoch}/{EPOCHS} | "
                    f"Step {step}/{steps_per_epoch} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"경과: {elapsed/60:.1f}분 | "
                    f"남은: {eta/60:.1f}분",
                    end="", flush=True,
                )

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch} 완료 | "
              f"평균 Loss: {total_loss/steps_per_epoch:.4f} | "
              f"소요: {epoch_time/60:.1f}분")

        evaluate(model, val_loader, device, epoch)
        save_checkpoint(model, tokenizer, optimizer, scheduler, epoch,
                        total_loss / steps_per_epoch)

    # 최종 모델 저장
    total_time = time.time() - total_start
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print(f"\n{'=' * 55}")
    print(f"  학습 완료!")
    print(f"  총 소요 시간: {total_time/60:.1f}분")
    print(f"  최종 모델 저장: {MODEL_SAVE_PATH}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    train()
