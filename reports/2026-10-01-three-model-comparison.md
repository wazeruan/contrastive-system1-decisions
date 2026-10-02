# Three-model tool-selection experiment — 2026-10-01

Source: user-provided completed pipeline results and external evaluation JSON. Remote checkpoints and scheduler artifacts were not independently downloaded or inspected. All three runs used seed 42, five epochs, FP32 and one Nibi H100 80GB; 128G host RAM was requested per stage.

## Architecture and optimizer

| Model | Scoring | Optimization | Approximate parameters |
|---|---|---|---:|
| Shared encoder | Shared DeBERTa; masked mean pooling; separate 768→256 heads; L2 normalization; dot product / 0.07 | AdamW, LR 2e-5 | 184.4M |
| Cross-encoder Muon | Joint query/tool input; CLS; 768→1 linear scoring head | Muon for hidden matrices, LR 2e-4; AdamW for embeddings/normalization/bias/output head, LR 2e-5 | 184M |
| Cross-encoder AdamW | Same joint input and scalar head | AdamW, LR 2e-5 | 184M |

Muon used momentum 0.95, five Newton–Schulz steps and `match_rms_adamw` adjustment. Weight decay was 0.01 and gradient clipping 1.0. Parameter counts are rounded model-card estimates, not measured checkpoint counts. Cross-encoders used a 448-token total pair limit; the shared encoder used 192/256 per-side limits. This is not a fully controlled interaction-only ablation.

## Accuracy comparison

| Dataset | Examples | Shared + AdamW | Cross + Muon/AdamW | Cross + AdamW |
|---|---:|---:|---:|---:|
| xLAM | 2,451 | 98.33% | 99.27% | 99.47% |
| BFCL live_multiple | 1,052 | 72.05% | 89.64% | 93.63% |
| when2call | 1,021 | 72.58% | 89.32% | 93.93% |
| toolace | 4,758 | 79.51% | 89.81% | 90.82% |

## Ranking and probability metrics

| Model | Dataset | MRR | NLL | Brier | ECE (15 bins) |
|---|---|---:|---:|---:|---:|
| Shared encoder + AdamW | xLAM | 0.9913 | 0.0718 | 0.0324 | 1.33% |
| Shared encoder + AdamW | BFCL live_multiple | 0.8415 | 0.7456 | 0.3897 | 6.52% |
| Cross-encoder + Muon/AdamW | xLAM | 0.9963 | 0.0225 | 0.0118 | 0.71% |
| Cross-encoder + Muon/AdamW | BFCL live_multiple | 0.9421 | 0.3060 | 0.1576 | 4.54% |
| Cross-encoder + AdamW | xLAM | 0.9971 | 0.0150 | 0.0080 | 0.26% |
| Cross-encoder + AdamW | BFCL live_multiple | 0.9654 | 0.1927 | 0.0991 | 1.56% |
| Shared encoder + AdamW | when2call | 0.8449 | 0.7382 | 0.3879 | 7.16% |
| Shared encoder + AdamW | toolace | 0.8857 | 0.5594 | 0.2956 | 3.34% |
| Cross-encoder + Muon/AdamW | when2call | 0.9412 | 0.3172 | 0.1639 | 4.44% |
| Cross-encoder + Muon/AdamW | toolace | 0.9445 | 0.2710 | 0.1456 | 1.08% |
| Cross-encoder + AdamW | when2call | 0.9670 | 0.1832 | 0.0949 | 1.43% |
| Cross-encoder + AdamW | toolace | 0.9462 | 0.2614 | 0.1320 | 2.04% |

## Paired external comparison

Rows count examples correct for only one of the two compared models, on identical frozen data.

| Left model | Right model | Dataset | Left only correct | Right only correct | Net gain for right |
|---|---|---|---:|---:|---:|
| Shared encoder + AdamW | Cross-encoder + Muon/AdamW | when2call | 51 | 222 | 171 |
| Shared encoder + AdamW | Cross-encoder + Muon/AdamW | toolace | 239 | 729 | 490 |
| Shared encoder + AdamW | Cross-encoder + AdamW | when2call | 25 | 243 | 218 |
| Shared encoder + AdamW | Cross-encoder + AdamW | toolace | 215 | 753 | 538 |
| Cross-encoder + Muon/AdamW | Cross-encoder + AdamW | when2call | 20 | 67 | 47 |
| Cross-encoder + Muon/AdamW | Cross-encoder + AdamW | toolace | 198 | 246 | 48 |

## Exploratory throughput

| Model | When2Call examples/s | ToolACE examples/s |
|---|---:|---:|
| Shared encoder + AdamW | 56.95 | 90.18 |
| Cross-encoder + Muon/AdamW | 63.17 | 75.12 |
| Cross-encoder + AdamW | 63.43 | 75.19 |

Timing includes tokenization and forward scoring, excludes checkpoint loading, and uses sequential model order. It is not a warmed latency benchmark, and shared-encoder tool caching was not measured. These numbers do not establish a universal speed ordering.

## Interpretation and limitations

- Cross-encoder AdamW has the highest accuracy on every reported slice. Its gains over the shared AdamW encoder are 21.58 percentage points on BFCL, 21.35 on When2Call and 11.31 on ToolACE.
- AdamW outperforms the tested Muon/AdamW settings in accuracy, but the ToolACE gain is only 1.01 percentage points. Muon/AdamW has lower ToolACE ECE (1.08% versus 2.04%). Do not claim optimizer dominance without tuning and multiple seeds.
- ToolACE is a synthetic training corpus repurposed as external evaluation; these project models were trained only on xLAM. When2Call is BFCL-derived, so it is correlated evidence, not an independent benchmark replication.
- Retained external examples have one distinct target and at least two original candidates. No-tool, clarification and multi-target cases were excluded. Exact normalized xLAM-query overlaps and within-source duplicate queries were filtered; this does not rule out near duplicates, schema overlap or pretraining contamination.
- xLAM schemas may overlap across splits. Typical external candidate sets contain about four tools. The large-candidate strata are too small to support scalability claims.
- These are custom tool-selection metrics, not argument-generation or official BFCL/When2Call benchmark scores. Existing xLAM calibration was reused without fitting to external labels.
- Next: three training seeds, paired uncertainty analysis, error inspection, controlled pooling/length ablations and cached-retrieval versus joint-reranking latency measurements. Retrieve independently then rerank jointly is a proposed deployment design; it has not been evaluated here.

## Records and sources

[Full-precision comparison](2026-10-01-three-model-comparison.json), [external results with frozen revisions/hashes/filter counts](2026-10-01-external-three-models.json), [original shared-encoder report](2026-09-30-shared-heads-h100.md).

Primary sources: [DeBERTa model card](https://huggingface.co/microsoft/deberta-v3-base), [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [When2Call](https://huggingface.co/datasets/nvidia/When2Call), [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).
