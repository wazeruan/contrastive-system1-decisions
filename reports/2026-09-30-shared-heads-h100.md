# Shared-backbone baseline results — 2026-09-30

Pipeline `20260930T193456Z-c784dfbd` was reported **SUCCEEDED** for training, calibration, xLAM evaluation, and BFCL evaluation. Source: user-provided `csd status` / `csd results` output; remote artifacts were not independently downloaded or inspected.

## Experiment

- Shared DeBERTa-v3-base backbone with separate query/action projection heads.
- Training code commit: `1c5e291`; config: `configs/shared-heads-h100.json`.
- Pipeline config SHA-256: `604608aa57032a08f4f9c4e723b5719133d575c4fb5eb3e5349cf8457ca4c721`.
- Nibi H100 80GB, FP32; requested host RAM: 128G per stage.
- Five epochs, 24,020 steps; best validation NLL: 0.07362169093509982; last validation accuracy: 98.09%.
- Temperature fitted on 1,269 calibration examples: 1.8731975555419922. Calibration NLL: 0.1301376223564148 → 0.09683699160814285.

## Evaluation

| Metric | xLAM test | BFCL live_multiple slice |
|---|---:|---:|
| Examples | 2,451 | 1,052 |
| Top-1 tool accuracy | 98.33% | 72.05% |
| MRR | 0.991262 | 0.841544 |
| NLL | 0.071802 | 0.745593 |
| Brier score | 0.032406 | 0.389725 |
| ECE (15 bins) | 1.33% | 6.52% |

Both evaluations used the fitted temperature and FP32. Full-precision reported values are in the [JSON record](2026-09-30-shared-heads-h100.json).

## Scope and limitations

These are custom candidate tool-selection metrics, not generated-argument accuracy or official BFCL leaderboard scores. xLAM tool schemas may overlap across splits. The 26.27 percentage-point accuracy gap accompanies different evaluation distributions and candidate sets; it does not identify a cause by itself. Exact dataset revisions, overlap audits, measured runtime/memory, and multi-seed uncertainty are not provided in this results record.

This is one baseline run. BM25, untrained-encoder, fully shared projection, and multiple-seed comparisons remain future work. Model weights and raw datasets are not included in this report.
