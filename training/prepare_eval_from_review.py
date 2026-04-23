"""
Build curated eval splits from the human-reviewed mismatch CSV.

Input:
    reports/eval_set_mismatches_review.csv

Outputs:
    data/eval/eval_candidates_curated.csv
    data/eval/eval_candidates_meta.csv
    data/eval/eval_candidates_boundary.csv
    data/eval/eval_candidates_drop.csv
    reports/eval_review_summary.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_CSV = PROJECT_ROOT / "reports" / "eval_set_mismatches_review.csv"
SOURCE_EVAL = PROJECT_ROOT / "data" / "eval" / "eval_candidates.csv"

CURATED_OUT = PROJECT_ROOT / "data" / "eval" / "eval_candidates_curated.csv"
META_OUT = PROJECT_ROOT / "data" / "eval" / "eval_candidates_meta.csv"
BOUNDARY_OUT = PROJECT_ROOT / "data" / "eval" / "eval_candidates_boundary.csv"
DROP_OUT = PROJECT_ROOT / "data" / "eval" / "eval_candidates_drop.csv"
SUMMARY_OUT = PROJECT_ROOT / "reports" / "eval_review_summary.json"


META_BUCKETS = {"meta_guideline"}
BOUNDARY_BUCKETS = {"severity_boundary"}
DROP_BUCKETS = {"drop"}
CURATED_BUCKETS = {
    "allow_conflict",
    "true_miss",
    "label_fix_normal",
    "label_fix_caution",
    "label_fix_suspicious",
}

LABEL_FIX_MAP = {
    "label_fix_normal": "정상",
    "label_fix_caution": "주의",
    "label_fix_suspicious": "의심",
}


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"Review CSV not found: {REVIEW_CSV}")
    if not SOURCE_EVAL.exists():
        raise FileNotFoundError(f"Source eval CSV not found: {SOURCE_EVAL}")

    review_rows = load_csv(REVIEW_CSV)
    eval_rows = load_csv(SOURCE_EVAL)

    review_by_id = {row["eval_id"]: row for row in review_rows}

    curated_rows: list[dict] = []
    meta_rows: list[dict] = []
    boundary_rows: list[dict] = []
    drop_rows: list[dict] = []
    unresolved_rows: list[dict] = []

    for row in eval_rows:
        review = review_by_id.get(row["eval_id"])
        if not review:
            curated_rows.append(row)
            continue

        human_bucket = (review.get("human_bucket") or "").strip()
        if not human_bucket:
            unresolved_rows.append(
                {
                    "eval_id": row["eval_id"],
                    "ad_text": row.get("ad_text", ""),
                    "reason": "needs_manual_review",
                }
            )
            continue

        if human_bucket in META_BUCKETS:
            meta_rows.append(row)
            continue

        if human_bucket in BOUNDARY_BUCKETS:
            boundary_rows.append(row)
            continue

        if human_bucket in DROP_BUCKETS:
            drop_rows.append(row)
            continue

        if human_bucket in CURATED_BUCKETS:
            fixed_row = dict(row)
            if human_bucket in LABEL_FIX_MAP:
                fixed_row["label"] = LABEL_FIX_MAP[human_bucket]
            curated_rows.append(fixed_row)
            continue

        unresolved_rows.append(
            {
                "eval_id": row["eval_id"],
                "ad_text": row.get("ad_text", ""),
                "reason": f"unknown_bucket:{human_bucket}",
            }
        )

    fieldnames = list(eval_rows[0].keys()) if eval_rows else []
    write_csv(CURATED_OUT, curated_rows, fieldnames)
    write_csv(META_OUT, meta_rows, fieldnames)
    write_csv(BOUNDARY_OUT, boundary_rows, fieldnames)
    write_csv(DROP_OUT, drop_rows, fieldnames)

    summary = {
        "source_eval": str(SOURCE_EVAL.relative_to(PROJECT_ROOT)),
        "review_csv": str(REVIEW_CSV.relative_to(PROJECT_ROOT)),
        "curated_out": str(CURATED_OUT.relative_to(PROJECT_ROOT)),
        "meta_out": str(META_OUT.relative_to(PROJECT_ROOT)),
        "boundary_out": str(BOUNDARY_OUT.relative_to(PROJECT_ROOT)),
        "drop_out": str(DROP_OUT.relative_to(PROJECT_ROOT)),
        "counts": {
            "source_eval_rows": len(eval_rows),
            "review_rows": len(review_rows),
            "curated_rows": len(curated_rows),
            "meta_rows": len(meta_rows),
            "boundary_rows": len(boundary_rows),
            "drop_rows": len(drop_rows),
            "unresolved_rows": len(unresolved_rows),
        },
        "human_bucket_counts": Counter(
            (row.get("human_bucket") or "").strip() or "EMPTY" for row in review_rows
        ),
        "unresolved": unresolved_rows[:200],
    }

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved -> {CURATED_OUT}")
    print(f"saved -> {META_OUT}")
    print(f"saved -> {BOUNDARY_OUT}")
    print(f"saved -> {DROP_OUT}")
    print(f"saved -> {SUMMARY_OUT}")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
