"""
Evaluate the current analysis pipeline on a fixed eval CSV.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.services.ad_domain_filter import predict_cosmetic
from analysis.services.ai_analyzer import analyze_with_kobert
from analysis.services.rule_engine import analyze_sentence


DEFAULT_INPUT = Path("data/eval/eval_candidates.csv")
DEFAULT_REPORT = Path("reports/eval_set_metrics.json")
DEFAULT_ERRORS = Path("reports/eval_set_mismatches.csv")
LABEL_ORDER = ["정상", "주의", "의심"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on eval_candidates.csv")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Eval CSV path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Metrics JSON output path")
    parser.add_argument("--errors", default=str(DEFAULT_ERRORS), help="Mismatch CSV output path")
    return parser.parse_args()


async def run_one(text: str) -> tuple[str, str, str, float]:
    cosmetic = predict_cosmetic(text)
    rule = analyze_sentence(text, force_cosmetic=False if cosmetic is None else cosmetic[0])
    final = await analyze_with_kobert(text, rule)
    return (
        rule.suspicion_level.value,
        final.suspicion_level.value,
        final.reason,
        final.score,
    )


async def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict(orient="records"):
        rule_label, final_label, reason, score = await run_one(str(row["ad_text"]))
        row["rule_label"] = rule_label
        row["pred_label"] = final_label
        row["pred_reason"] = reason
        row["pred_score"] = score
        row["is_correct"] = final_label == row["label"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    report_path = Path(args.report)
    errors_path = Path(args.errors)

    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str)
    results = asyncio.run(evaluate(df))

    y_true = results["label"].tolist()
    y_pred = results["pred_label"].tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    acc = accuracy_score(y_true, y_pred)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "rows": len(results),
        "accuracy": acc,
        "label_order": LABEL_ORDER,
        "confusion_matrix": matrix.tolist(),
        "report": report,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    mismatches = results[~results["is_correct"]].copy()
    mismatches.to_csv(errors_path, index=False, encoding="utf-8-sig")

    print(f"saved report -> {report_path}")
    print(f"saved mismatches -> {errors_path}")
    print(f"rows -> {len(results)}")
    print(f"accuracy -> {acc:.4f}")
    print("label counts:")
    print(results["label"].value_counts().to_string())
    print("prediction counts:")
    print(results["pred_label"].value_counts().to_string())
    print(f"mismatches -> {len(mismatches)}")


if __name__ == "__main__":
    main()
