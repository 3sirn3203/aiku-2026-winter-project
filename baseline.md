# LLM-Based Tabular Data Analysis Agent — Baseline Draft

Date: 2026-01-27
Owner: Project Team
Status: Draft v0.1 (baseline)

## 1) Objective
Build an autonomous agent that, given a tabular dataset (CSV/Excel), repeatedly performs:
- Data exploration
- Hypothesis generation
- Feature engineering
- Validation (cross-validation)

The agent outputs reusable Python artifacts:
- `pipeline.py` (data cleaning + feature engineering)
- `train_advanced.py` (final training script for a higher-capacity model)
- `inference.py` (prediction pipeline)
- A lightweight report (metrics, features tried, best config)

## 2) Constraints & Non-Goals
- Primary data type: tabular data only (CSV/Excel)
- Baseline proxy modeling: lightweight models (scikit-learn, XGBoost) for fast iteration
- No GUI required in baseline
- No external data augmentation (first iteration)

## 3) High-Level Architecture

### Directories (baseline contract)
- `configs/`: YAML configs for model, prompts, and runtime paths
- `data/`: raw/processed/submissions (data directory is gitignored)
- `generated/`: agent-generated scripts + reports
- `src/`: core agent logic (LangGraph + LLM wrapper + sandbox + utils)
- `main.py`: entry point to run the agent workflow

### Key Components
1) **LLM Wrapper (`src/llm/`)**
   - Google AI Studio (Gemini) via `google-generativeai`
   - Deterministic settings for baseline (temperature <= 0.2)

2) **Agent State (`src/agent/state.py`)**
   - TypedDict for agent state
   - Minimum fields:
     - `input_file`, `target_column`, `problem_type`
     - `plan`, `generated_code`, `execution_result`
     - `history`, `metrics`, `best_config`

3) **LangGraph Workflow (`src/agent/graph.py`)**
   - Nodes: `plan_step` -> `code_gen_step` -> `execute_step` -> `review_step`
   - Loop until `max_iters` or convergence

4) **Sandbox (`src/sandbox/`)**
   - Execute generated code in a controlled environment (Docker / local)
   - Baseline: local execution with guardrails (timeouts, filesystem boundaries)

5) **Utils (`src/utils/`)**
   - Logging
   - File IO and safe path helpers

## 4) Agent Loop (Baseline)

### Step A: Planning
- Summarize dataset schema (columns, dtypes, missing values)
- Identify target type (classification/regression)
- Propose feature engineering ideas
- Define an evaluation plan (metric, CV folds)

### Step B: Code Generation
- Draft `pipeline.py` with:
  - missing value handling
  - encoding of categorical variables
  - basic numerical transforms
- Draft a proxy model training script to evaluate features

### Step C: Execution
- Run pipeline + proxy training
- Collect metrics and error logs

### Step D: Feedback / Review
- Decide whether to iterate (based on metric gain or errors)
- Update plan and propose next modifications

## 5) Output Contracts

### 5.1 `generated/scripts/pipeline.py`
- `build_preprocess_pipeline(df, target_col) -> (X, y, preprocess_obj)`
- Handles:
  - Missing values
  - Categorical encoding
  - Train/test split (optional)

### 5.2 `generated/scripts/train_advanced.py`
- Uses best feature set from proxy loop
- Placeholder for higher-capacity model (e.g., ensemble or DL)
- Saves model artifact to `data/submissions/`

### 5.3 `generated/scripts/inference.py`
- Loads trained model
- Applies preprocessing
- Generates predictions

### 5.4 `generated/reports/summary.md`
- Dataset summary
- Best proxy score
- Selected feature transformations
- Notes about next iteration

## 6) Configuration (baseline schema)

### `configs/config.yaml`
- `paths`:
  - `data_raw`, `data_processed`, `generated_scripts`, `generated_reports`
- `agent`:
  - `max_iters`, `timeout_sec`
- `model`:
  - `proxy_type`: `xgboost` or `sklearn`
  - `metric`: e.g., `rmse` / `auc`

### `configs/prompts.yaml`
- Planner: dataset understanding + FE hypothesis
- Coder: produce runnable pandas/sklearn code
- Reviewer: parse logs + decide next action

## 7) Minimal Runtime Flow (`main.py`)
1) Load config
2) Initialize LLM + agent state
3) Build LangGraph and run
4) Save artifacts to `generated/`

## 8) Risk & Limitations (baseline)
- Generated code may fail on corner-case schemas
- Model selection is coarse-grained (proxy only)
- No automatic unit tests for generated scripts (first iteration)

## 9) Next Steps (post-baseline)
- Add automatic test runner for generated pipeline
- Add schema validation and data drift checks
- Expand proxy model set (LightGBM, CatBoost)
- Track experiments and iterations in a structured log
