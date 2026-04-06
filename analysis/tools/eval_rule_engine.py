"""
Rule engine quick evaluation on the clean sheet.

Usage:
    python -m analysis.tools.eval_rule_engine --limit 1000

Reads `data/merged/merged_clean.csv`, runs the rule engine on each row,
and prints simple accuracy/recall stats. Meant for regression checks,
not as a formal benchmark.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from analysis.services.rule_engine import analyze_sentence
from analysis.models.schemas import SuspicionLevel


LABEL_MAP = {
    "의심": SuspicionLevel.SUSPICIOUS,
    "주의": SuspicionLevel.CAUTION,
    "정상": SuspicionLevel.NORMAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule engine against clean sheet.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/merged/merged_clean.csv"),
        help="Path to merged clean CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of rows to sample from the top (0 = use all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = []
    with args.csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            rows.append(row)

    conf = defaultdict(Counter)  # expected -> predicted -> count
    total = 0
    for row in rows:
        label_str = (row.get("의심도 라벨") or "").strip()
        expected = LABEL_MAP.get(label_str)
        if expected is None:
            continue  # skip NA rows
        sentence = row.get("정제 문구") or ""
        result = analyze_sentence(sentence)
        conf[expected][result.suspicion_level] += 1
        total += 1

    if total == 0:
        raise SystemExit("No rows evaluated (check labels in CSV).")

    print(f"Evaluated rows: {total} (limit={args.limit or 'all'})")
    print()
    levels = [SuspicionLevel.NORMAL, SuspicionLevel.CAUTION, SuspicionLevel.SUSPICIOUS]

    def _safe_div(num: int, den: int) -> float:
        return round(num / den, 3) if den else 0.0

    # Per-class recall and precision
    for lvl in levels:
        tp = conf[lvl][lvl]
        fn = sum(conf[lvl][p] for p in levels if p != lvl)
        fp = sum(conf[other][lvl] for other in levels if other != lvl)
        recall = _safe_div(tp, tp + fn)
        precision = _safe_div(tp, tp + fp)
        print(f"[{lvl.name}] precision={precision} recall={recall} tp={tp} fp={fp} fn={fn}")

    # Overall accuracy
    correct = sum(conf[l][l] for l in levels)
    acc = _safe_div(correct, total)
    print(f"\nOverall accuracy: {acc} ({correct}/{total})")


if __name__ == "__main__":
    main()
