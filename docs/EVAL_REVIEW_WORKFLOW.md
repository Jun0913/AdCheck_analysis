# Eval Review Workflow

이 문서는 `reports/eval_set_mismatches_review.csv`를 사람이 검수할 때 따르는 최소 절차를 정리한다.

## 입력 파일

- `reports/eval_set_mismatches_review.csv`

주요 컬럼:
- `ad_text`
- `label`
- `rule_label`
- `pred_label`
- `pred_reason`
- `review_bucket`
- `review_note`
- `suggested_action`
- `human_bucket`
- `human_note`

## 사람이 해야 하는 일

1. `ad_text`를 읽고 실제 광고 문구인지 판단
2. 현재 `label`이 유지돼야 하는지 확인
3. `review_bucket` 제안이 맞는지 보고 `human_bucket`에 최종 확정
4. 이유를 `human_note`에 짧게 적기

## `human_bucket` 권장값

- `meta_guideline`
- `allow_conflict`
- `true_miss`
- `severity_boundary`
- `label_fix_normal`
- `label_fix_caution`
- `label_fix_suspicious`
- `drop`

## 빠른 판단 기준

### `meta_guideline`
- 규정 설명문, 금지사례 해설문, 지침 문장
- 실제 서비스 입력 광고문구로 보기 어려움

### `allow_conflict`
- 허용 예외 문구 또는 조건부 허용 문구
- 현재 시스템만 과하게 잡고 있음

### `true_miss`
- 실제 광고문구인데 시스템이 놓쳤음
- 추가 패턴/룰 후보

### `severity_boundary`
- 광고문구는 맞음
- 다만 `주의/의심` 강도만 다름

### `label_fix_*`
- 현재 원본 `label` 자체를 바꾸는 것이 맞다고 판단될 때 사용

### `drop`
- 실제 평가셋에서 제외하는 것이 맞는 문장
- 중복, 불필요한 단어 조각, 평가 목적과 맞지 않는 항목

## 검수 완료 후

검수가 끝나면 아래 스크립트로 정제 eval 셋을 생성한다.

```bash
python training/prepare_eval_from_review.py
```

산출물:
- `data/eval/eval_candidates_curated.csv`
- `data/eval/eval_candidates_meta.csv`
- `data/eval/eval_candidates_boundary.csv`
- `data/eval/eval_candidates_drop.csv`
- `reports/eval_review_summary.json`

## 주의

- `human_bucket`이 비어 있으면 해당 행은 `needs_manual_review`로 분류된다.
- `meta_guideline`은 메인 eval에서 제외하는 것이 기본 권장이다.
- `label_fix_*`를 사용했다면 이후 학습 데이터에도 같은 기준을 반영해야 한다.
