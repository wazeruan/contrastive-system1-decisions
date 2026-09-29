# Contrastive System-1 Decisions

A research prototype for scoring a variable set of typed tool/action candidates against a query. The main model reuses one DeBERTa encoder for query and candidate schemas, with separate small projection heads. A fully tied projection is the primary sharing ablation; separate encoders are an optional capacity ablation.

“Jev-like” here names the proposed fast candidate-selection role. The implementation is an independent contrastive-learning experiment; it does not claim to reproduce Jev's unreleased model internals.

## Dataset and benchmark

- **Training:** [Salesforce APIGen / xLAM Function-Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k). Rows contain a query, tool schemas, and labeled tool calls. The adapter keeps examples with one distinct target tool and at least two candidate schemas; parallel/multi-tool calls are excluded from this first single-choice objective. APIGen reports format, execution, and semantic checks on its generated data. Its license is CC BY 4.0, and the Hugging Face files are gated: accept the dataset conditions in your account before downloading. Cite APIGen and retain attribution if you share derived artifacts.
- **Primary external benchmark:** [BFCL's single-turn live `multiple` category](https://github.com/EnlightenedAI/BFCL/blob/main/berkeley-function-call-leaderboard/bfcl_eval/data/README.md), available in the public snapshot as `BFCL_v3_live_multiple`. Its user-contributed cases ask the model to choose among several function schemas. BFCL changes over time; the data preparer records the exact resolved snapshot revision and observed example count.
- **Secondary check:** BFCL `multiple` / `BFCL_v3_multiple`, the controlled single-turn multiple-function category.

The model returns a tool ID and a conditional choice distribution. It does **not** generate arguments or execute a tool. Report the benchmark result as **custom tool-selection accuracy/MRR**, not as an official BFCL score: BFCL's official score also evaluates full calls and arguments. BFCL V2 Live `irrelevance` is a useful future abstention set, but this first training adapter has no trusted no-tool labels, so it does not claim abstention performance. Keep all BFCL rows out of training and hyperparameter selection. The APIGen authors report BFCL evaluations, so audit exact/near-duplicate queries and tool-schema overlap before interpreting the external score as fully novel.

The xLAM adapter groups exact normalized duplicate queries before splitting into train/validation/calibration/test (80/5/5/10). Tool schemas can occur in multiple splits; the manifest states this limitation. Report seen-tool and unseen-tool-family results separately if you add an API-family holdout.

## Model and objective

For each example, the **same encoder with the same backbone weights** reads `[QUERY] <request>` and, in a separate forward pass, each `[ACTION] <canonical tool schema>`. Mean-pool the outputs, apply role-specific 256-dimensional projection heads, L2-normalize, and score with cosine similarity divided by `tau=0.07`. There is one transformer encoder in the main model, not a query encoder plus a separately parameterized action encoder. Candidate-set cross entropy trains the probability of selecting an action given the supplied candidates; it is not a real-world action-success probability.

The three sharing choices answer different questions:

- `shared_heads` (**main model**): one shared backbone with separate query/action projection heads. It preserves common language features while allowing the two roles to map into the comparison space differently.
- `shared_tied` (**main ablation**): the same single backbone and one shared projection for both roles. It tests whether role prefixes alone are enough.
- `separate` (**optional ablation only**): independently fine-tuned query and action backbones with separate heads. It tests whether role-specific capacity helps enough to justify roughly twice the backbone parameters and more compute; it is excluded from the minimal experiment below.

For query group `i` and its supplied candidates `j`, the training loss is

`L_i = -sum_j y_ij log softmax_j(cos(q_i, a_ij) / tau)`,

where `y_i` is one-hot for the single target tool. Scores are normalized only over that query's candidate list; candidates from other examples are not silently treated as negatives.

## Concrete data and training recipe

The source rows contain `query`, `tools`, and `answers`. The preparation step keeps one distinct target tool with at least two candidate schemas, writes normalized JSONL, and records filtering counts and source revision. A normalized row has this shape:

```json
{"example_id":"xlam-17","query":"[QUERY] find flights to Edmonton","candidates":[{"candidate_id":"search_flights","text":"[ACTION] {\"name\":\"search_flights\",...}"},{"candidate_id":"book_hotel","text":"[ACTION] {\"name\":\"book_hotel\",...}"}],"target_weights":[1.0,0.0],"group_id":"<normalized-query-sha256>","source":"Salesforce/xlam-function-calling-60k"}
```

Exact normalized duplicate queries stay in the same 80/5/5/10 train/validation/calibration/test split. Each batch has four query groups; the model scores each query only against that row's candidates, with action strings flattened for encoding. The checked-in profile uses DeBERTa-v3-base, 256-dimensional normalized vectors, `tau=0.07`, BF16, AdamW at `2e-5`, weight decay `0.01`, gradient clipping at `1.0`, query/action token caps of 192/256, gradient checkpointing, up to five epochs, and validation-NLL early stopping.

Use the xLAM-supplied alternatives as the main condition. For a separate hard-negative ablation, `--bm25-negatives 2` appends up to two BM25-retrieved schemas drawn only from the training fold. They are weak negatives because some may be valid but unlabeled tools; inspect this condition separately. Never mine from validation, calibration, test, or BFCL data.

At inference, pass the request and exact candidate schemas, compute one score per candidate, return the top-ranked `candidate_id`, and optionally return the candidate-set softmax distribution. The calibration stage fits one scalar temperature by NLL on the calibration split only; evaluation applies it to the untouched test or BFCL selector examples. This prototype selects a tool and does not fill arguments or execute actions.

The default model is Microsoft's 184M-parameter DeBERTa-v3-base, pinned to a repository revision and licensed MIT. The initial conservative profile uses BF16 on one H100, AdamW, four query groups per batch, gradient checkpointing, and early stopping on validation NLL. `configs/` contains the three architecture variants with the same data/model/training settings. The separate-encoder model has roughly twice the backbone parameters; report quality, memory, and latency together. Increase the batch only after an allocated-node pilot.

## Local project setup

This is an SSH-first, headless project. No GUI or notebook is required. Place the checkout, model cache, and outputs on storage visible to both CPU and H100 nodes; the CPU download stages and GPU training stages share those paths. If the cluster provides preferred shared cache locations, set `HF_HOME` and optionally `CSD_MODEL_CACHE_MANIFEST` in the shared environment script so the preparation and GPU jobs use the same paths. The manifest variable can be a project-relative or absolute path and is included in submission receipts. For local development, prepare an isolated environment with:

```bash
uv sync
```

After accepting the gated xLAM conditions and logging in to Hugging Face, set up the environment in a CPU allocation, then prepare data in separate CPU jobs. The Slurm account, CPU/GPU partitions, and exact GPU request syntax are site-specific, so supply them at launch:

```bash
./scripts/submit.sh --stage setup --account "$SLURM_ACCOUNT" --partition "$CPU_PARTITION"
./scripts/submit.sh --stage prepare-xlam --account "$SLURM_ACCOUNT" --partition "$CPU_PARTITION" \
  --dependency SETUP_JOB_ID --bm25-negatives 0
./scripts/submit.sh --stage prepare-bfcl --account "$SLURM_ACCOUNT" --partition "$CPU_PARTITION" \
  --dependency SETUP_JOB_ID
./scripts/submit.sh --stage prepare-model --account "$SLURM_ACCOUNT" --partition "$CPU_PARTITION" \
  --config configs/shared-heads-h100.json --dependency SETUP_JOB_ID
```

Replace the uppercase placeholders with the cluster's confirmed values and each preceding command's printed `JOB_ID`. `prepare-model` downloads the pinned DeBERTa weights in a CPU allocation; GPU jobs then run with Hugging Face offline mode and require the matching cache manifest. If the cluster exposes Python/uv through modules, provide a shared shell setup file with `--env-script PATH`; it is sourced on the submit and compute nodes. For model stages, also pass the confirmed GPU resource syntax, for example `--gpu-resource 'gpu:h100:1'` only if your Slurm site uses that form. Run `--stage preflight` after setup to confirm the actual H100 allocation before a long training job. `HF_TOKEN` may be present in the submitting environment for the gated xLAM download; the wrapper forwards the environment without printing the token. Never put it in a command line, config file, Slurm log, or Git. You must accept the dataset conditions yourself. The BFCL snapshot is public and Apache-2.0 licensed.

## Train and evaluate

Once setup and data jobs have completed, run the three architecture variants. Each run directory is claimed atomically; if a requested name exists, the wrapper uses the next numeric suffix. Use `--dependency JOB_ID` to connect stages with `afterok` semantics when an input is still being produced by another Slurm job.

```bash
./scripts/submit.sh --stage preflight --account "$SLURM_ACCOUNT" --partition "$H100_PARTITION" \
  --gpu-resource "$H100_GPU_REQUEST" --dependency MODEL_CACHE_JOB_ID

./scripts/submit.sh --stage train --account "$SLURM_ACCOUNT" --partition "$H100_PARTITION" \
  --gpu-resource "$H100_GPU_REQUEST" --config configs/shared-heads-h100.json \
  --data-dir XLAM_RUN_DIR --dependency MODEL_CACHE_JOB_ID,XLAM_JOB_ID

./scripts/submit.sh --stage calibrate --account "$SLURM_ACCOUNT" --partition "$H100_PARTITION" \
  --gpu-resource "$H100_GPU_REQUEST" --checkpoint RUN_DIR/checkpoints/best.pt \
  --data XLAM_RUN_DIR/calibration.jsonl --dependency TRAIN_JOB_ID

./scripts/submit.sh --stage evaluate --account "$SLURM_ACCOUNT" --partition "$H100_PARTITION" \
  --gpu-resource "$H100_GPU_REQUEST" --checkpoint RUN_DIR/checkpoints/best.pt \
  --data XLAM_RUN_DIR/test.jsonl --calibration CALIBRATION_RUN_DIR/calibration.json \
  --dependency CALIBRATION_JOB_ID

./scripts/submit.sh --stage benchmark --account "$SLURM_ACCOUNT" --partition "$H100_PARTITION" \
  --gpu-resource "$H100_GPU_REQUEST" --checkpoint RUN_DIR/checkpoints/best.pt \
  --data-dir BFCL_DATA_DIR --category live_multiple --dependency BFCL_PREP_JOB_ID
```

Replace `XLAM_RUN_DIR`, `BFCL_DATA_DIR`, `RUN_DIR`, `CALIBRATION_RUN_DIR`, and job ID placeholders with the printed values. Training can be resumed only with an explicit `--resume --output-dir EXISTING_RUN_DIR`; it verifies the saved config and checkpoint. Use `--stage evaluate` for held-out scoring/inference and `--stage preflight` for GPU debugging. Slurm jobs write a durable receipt, timestamped stdout/stderr logs, and atomic state markers. The submit wrapper checks startup for a bounded interval and reports `STARTED`, `PENDING`, `SUCCEEDED`, `PREEMPTED`, `FAILED`, `RUNNING_STARTUP_UNVERIFIED`, or `UNKNOWN/UNVERIFIED`; it leaves pending jobs in place. Do not run training, model downloads, or dataset preparation directly on a login node.

The allocation preflight requires an H100 with at least 75 GiB reported device memory and BF16 support. GPU visibility on a login node is not accepted as evidence. The initial profile uses four query groups per batch, sequence caps of 192/256, and gradient checkpointing; adjust only after an allocated-node pilot. Training saves atomic `last.pt` and validation-selected `best.pt` checkpoints, writes epoch metrics, and handles `USR1`, `TERM`, and `INT` by finishing the current epoch and checkpointing.

## Planned ablations

1. Encoder sharing: compare `shared_tied` with the one-backbone `shared_heads` model using the same checkpoint, split, seed, and optimization budget. Keep `separate` for an optional follow-up capacity ablation.
2. Negative set: provided candidate schemas versus two training-fold-only BM25-retrieved weak negatives (`--bm25-negatives 2`). No validation/calibration/test tool schemas are mined. Retrieved tools may be valid but unlabeled actions, so report this as a noisy-negative ablation and audit sampled cases before interpreting it.
3. Calibration: no calibration versus one scalar temperature fitted only on calibration data.
4. Evaluation: top-1 tool accuracy, MRR, NLL, Brier score, 15-bin ECE, mean candidate count, and latency by candidate-set size. Repeat each architecture with at least three seeds.

## Minimal MSc experiment

1. Run a short shared-head pilot on the full filtered training split to confirm allocation startup, memory, checkpointing, and data sizes; do not tune on BFCL.
2. Compare the two unified-encoder variants, `shared_tied` and `shared_heads`, using the same xLAM split, optimizer budget, and three seeds (42, 43, 44). Report mean and standard deviation for validation-selected test metrics; report parameter count, peak allocated GPU memory, and per-query latency alongside quality.
3. Fit one scalar temperature per seed on the calibration split. Evaluate once on the untouched xLAM test split and BFCL V2 Live `multiple`; use the controlled BFCL V3 `multiple` set as the smaller secondary check.
4. For the negative-set ablation, compare native candidates against BM25 augmentation on the strongest architecture. Keep that comparison separate from the architecture claim and inspect the retrieved negatives for false-negative rate.

Configs are checked in for seeds 42, 43, and 44 for each architecture. The submission wrapper snapshots the selected config into each run directory and records its hash. The project does not download data or submit jobs until you invoke the corresponding commands on the cluster.

## Data and model citations

- APIGen: Liu et al., [Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518).
- BFCL: [Berkeley Function Calling Leaderboard data and category definitions](https://github.com/EnlightenedAI/BFCL/blob/main/berkeley-function-call-leaderboard/bfcl_eval/data/README.md).
- Encoder: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base).
- Calibration: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
