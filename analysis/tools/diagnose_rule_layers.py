"""
Rule-layer diagnostics for merged_clean.

Usage:
    python -m analysis.tools.diagnose_rule_layers
    python -m analysis.tools.diagnose_rule_layers --limit 5000

Outputs:
    reports/rule_layer_diagnostics.json
    reports/rule_layer_missed_risky.csv
    reports/rule_layer_overflagged_normal.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from analysis.models.schemas import SuspicionLevel
from analysis.services.rule_engine import analyze_sentence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "merged" / "merged_clean.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "rule_layer_diagnostics.json"
DEFAULT_MISSED_CSV = PROJECT_ROOT / "reports" / "rule_layer_missed_risky.csv"
DEFAULT_OVERFLAGGED_CSV = PROJECT_ROOT / "reports" / "rule_layer_overflagged_normal.csv"

TEXT_FIELD_CANDIDATES = ("정제 문구",)
LABEL_FIELD_CANDIDATES = ("의심도 라벨",)
SOURCE_FIELD_CANDIDATES = ("source_type", "출처", "??? ??")
HUMAN_FIELD_CANDIDATES = ("human_reviewed", "???? ??")

LABEL_MAP = {
    "정상": SuspicionLevel.NORMAL,
    "주의": SuspicionLevel.CAUTION,
    "의심": SuspicionLevel.SUSPICIOUS,
}

LEVEL_ORDER = {
    SuspicionLevel.NORMAL: 0,
    SuspicionLevel.CAUTION: 1,
    SuspicionLevel.SUSPICIOUS: 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose rule-layer behavior on merged_clean.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows (0 = all).")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT, help="Output summary JSON path.")
    parser.add_argument("--missed-csv", type=Path, default=DEFAULT_MISSED_CSV, help="Output CSV for risky rows missed by rules.")
    parser.add_argument("--overflagged-csv", type=Path, default=DEFAULT_OVERFLAGGED_CSV, help="Output CSV for normal rows over-flagged by rules.")
    parser.add_argument("--sample-limit", type=int, default=200, help="Maximum rows to export per candidate CSV.")
    return parser.parse_args()


def resolve_field(fieldnames: Iterable[str] | None, candidates: tuple[str, ...]) -> str | None:
    if not fieldnames:
        return None
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def serialize_counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def level_value(level: SuspicionLevel) -> str:
    return level.value


def write_candidate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "text",
        "expected_label",
        "predicted_label",
        "score",
        "matched_patterns",
        "matched_keywords",
        "reason",
        "source",
        "human_reviewed",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows_evaluated = 0
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    pattern_counts: Counter[str] = Counter()
    pattern_by_expected: dict[str, Counter[str]] = defaultdict(Counter)
    no_pattern_by_expected: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_by_expected: dict[str, Counter[str]] = defaultdict(Counter)
    missed_risky: list[dict[str, object]] = []
    overflagged_normal: list[dict[str, object]] = []
    risky_without_patterns: list[dict[str, object]] = []

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        text_field = resolve_field(reader.fieldnames, TEXT_FIELD_CANDIDATES)
        label_field = resolve_field(reader.fieldnames, LABEL_FIELD_CANDIDATES)
        source_field = resolve_field(reader.fieldnames, SOURCE_FIELD_CANDIDATES)
        human_field = resolve_field(reader.fieldnames, HUMAN_FIELD_CANDIDATES)

        if not text_field or not label_field:
            raise SystemExit(f"Required columns not found. fields={reader.fieldnames}")

        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break

            text = (row.get(text_field) or "").strip()
            label_str = (row.get(label_field) or "").strip()
            expected = LABEL_MAP.get(label_str)
            if not text or expected is None:
                continue

            source = (row.get(source_field) or "") if source_field else ""
            human_reviewed = (row.get(human_field) or "") if human_field else ""

            result = analyze_sentence(text)

            rows_evaluated += 1
            label_counts[label_str] += 1
            prediction_counts[level_value(result.suspicion_level)] += 1
            confusion[label_str][level_value(result.suspicion_level)] += 1
            source_counts[source] += 1
            source_by_expected[label_str][source] += 1

            if result.matched_patterns:
                for pattern in result.matched_patterns:
                    pattern_counts[pattern] += 1
                    pattern_by_expected[label_str][pattern] += 1
            else:
                no_pattern_by_expected[label_str] += 1

            candidate_row = {
                "text": text,
                "expected_label": label_str,
                "predicted_label": level_value(result.suspicion_level),
                "score": round(result.score, 3),
                "matched_patterns": "|".join(result.matched_patterns),
                "matched_keywords": "|".join(result.matched_keywords[:8]),
                "reason": result.reason,
                "source": source,
                "human_reviewed": human_reviewed,
            }

            if LEVEL_ORDER[expected] > LEVEL_ORDER[result.suspicion_level]:
                if len(missed_risky) < args.sample_limit:
                    missed_risky.append(candidate_row)
                if not result.matched_patterns and len(risky_without_patterns) < args.sample_limit:
                    risky_without_patterns.append(candidate_row)

            if expected == SuspicionLevel.NORMAL and result.suspicion_level != SuspicionLevel.NORMAL:
                if len(overflagged_normal) < args.sample_limit:
                    overflagged_normal.append(candidate_row)

    if rows_evaluated == 0:
        raise SystemExit("No rows evaluated.")

    accuracy = ratio(
        sum(confusion[label][label] for label in LABEL_MAP if label in confusion),
        rows_evaluated,
    )

    summary = {
        "dataset": str(args.csv.relative_to(PROJECT_ROOT) if args.csv.is_relative_to(PROJECT_ROOT) else args.csv),
        "rows_evaluated": rows_evaluated,
        "accuracy_rule_only": accuracy,
        "label_distribution": {
            "counts": serialize_counter(label_counts),
            "ratios": {label: ratio(count, rows_evaluated) for label, count in label_counts.items()},
        },
        "prediction_distribution": {
            "counts": serialize_counter(prediction_counts),
            "ratios": {label: ratio(count, rows_evaluated) for label, count in prediction_counts.items()},
        },
        "confusion": {
            expected: serialize_counter(preds)
            for expected, preds in confusion.items()
        },
        "pattern_summary": {
            "overall": serialize_counter(pattern_counts),
            "by_expected_label": {
                expected: serialize_counter(counter)
                for expected, counter in pattern_by_expected.items()
            },
            "no_pattern_by_expected_label": serialize_counter(no_pattern_by_expected),
        },
        "source_summary": {
            "overall": serialize_counter(source_counts),
            "by_expected_label": {
                expected: serialize_counter(counter)
                for expected, counter in source_by_expected.items()
            },
        },
        "candidate_counts": {
            "missed_risky": len(missed_risky),
            "overflagged_normal": len(overflagged_normal),
            "risky_without_patterns": len(risky_without_patterns),
        },
        "top_examples": {
            "missed_risky": missed_risky[:20],
            "overflagged_normal": overflagged_normal[:20],
            "risky_without_patterns": risky_without_patterns[:20],
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_candidate_csv(args.missed_csv, missed_risky)
    write_candidate_csv(args.overflagged_csv, overflagged_normal)

    print(f"Evaluated: {rows_evaluated}")
    print(f"Rule-only accuracy: {accuracy}")
    print(f"Report: {args.report_json}")
    print(f"Missed risky CSV: {args.missed_csv} ({len(missed_risky)} rows)")
    print(f"Over-flagged normal CSV: {args.overflagged_csv} ({len(overflagged_normal)} rows)")


if __name__ == "__main__":
    main()
