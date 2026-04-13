"""
Evaluate a trained KoBERT classifier on the fixed validation split and
save metrics for future comparisons.

This reuses preprocess/AdDataset from training.train so the split and
label mapping stay identical to training.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Reuse the exact data handling used during training
from train import preprocess, AdDataset, MAX_LEN, BATCH_SIZE, LABEL_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KoBERT ad classifier")
    parser.add_argument(
        "--model-dir",
        default="models/kobert_ad_classifier",
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--data",
        default="data/merged/merged_clean.csv",
        help="Merged training CSV (same as training input)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split (must match training for fairness)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio (keep identical to training split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Eval batch size",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_LEN,
        help="Tokenizer max length",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output metrics JSON path (default: reports/metrics_<timestamp>.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console metric printout (useful when called from training)",
    )
    return parser.parse_args()


def make_output_path(path: str | None) -> str:
    if path:
        return path
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("reports", f"metrics_{ts}.json")


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_and_save(
    model_dir: str = "models/kobert_ad_classifier",
    data: str = "data/merged/merged_clean.csv",
    seed: int = 42,
    test_size: float = 0.2,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    output: str | None = None,
    quiet: bool = False,
):
    """
    Evaluate a classification model and persist metrics as JSON.
    Returns (summary_dict, output_path).
    """
    set_seed(seed)

    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found: {data}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load data and create the same split used during training
    df_raw = pd.read_csv(data, encoding="utf-8-sig", dtype=str)
    df = preprocess(df_raw)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=len(LABEL_NAMES)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_dataset = AdDataset(val_df["text"], val_df["label"], tokenizer, max_len)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device.type == "cuda" else 0,
    )

    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    report = classification_report(
        true_labels,
        preds,
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )
    acc = accuracy_score(true_labels, preds)

    out_path = make_output_path(output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = {
        "model_dir": model_dir,
        "data": data,
        "seed": seed,
        "test_size": test_size,
        "timestamp": datetime.now().isoformat(),
        "accuracy": acc,
        "report": report,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not quiet:
        print(f"\nSaved metrics to {out_path}")
        print(f"Overall accuracy: {acc:.4f}")
        for label in LABEL_NAMES:
            m = report[label]
            print(
                f"{label}: precision {m['precision']:.3f} | recall {m['recall']:.3f} | f1 {m['f1-score']:.3f} | support {m['support']}"
            )

    return summary, out_path


def main():
    args = parse_args()
    evaluate_and_save(
        model_dir=args.model_dir,
        data=args.data,
        seed=args.seed,
        test_size=args.test_size,
        batch_size=args.batch_size,
        max_len=args.max_len,
        output=args.output,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
