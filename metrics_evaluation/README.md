# Metrics Evaluation

Evaluation pipeline for autonomous coding agents (Master's thesis).
Evaluates agent trajectories using deterministic metrics and LLM-as-a-Judge.

---

## Directory Structure

```
metrics_evaluation/
├── metrics_evaluation.py         # Main evaluation script (single trajectory)
├── batch_evaluation.py           # Batch processing of all issues
├── consolidate_results.py        # Consolidates eval_*.json into consolidated_results.csv
├── evaluation_data_models.py     # Data classes: StandardStep, EvaluationTrace
├── evaluation_prompts.py         # LLM-Judge prompts (current version)
├── evaluation_prompts_v1.py      # Backup: original prompts
├── evaluation_prompts_v2.py      # Alternative prompts version
├── review_trace.py               # Helper script for displaying trajectories
│
├── agent_parsers/
│   ├── openhands_parser.py       # Parser for OpenHands JSON logs
│   ├── sweagent_parser.py        # Parser for SWE-Agent .traj files
│   ├── metagpt_parser.py         # Parser for MetaGPT .txt/.log files
│   └── live_sweagent_parser.py   # Parser for Live-SWE-Agent
│
├── descriptive_analysis.py       # Descriptive analysis + plots (RQ1, RQ3)
├── annotation_analysis.py        # Visualization of manual annotations (RQ2)
├── spearman_correlation.py       # Spearman correlation auto vs. human (RQ2)
├── kappa_sample_rate_comparison.py  # Weighted Kappa: SR stability + Cross-Model (RQ4, RQ5)
├── auc_predictor_analysis.py     # AUC predictor analysis (RQ4.3)
├── plot_spearman_sr_rq4.py       # Spearman SR=1 vs. SR=5 comparison (RQ4.2)
│
├── manual_annotations.csv        # Manual expert ratings
├── Labeling_Guide.md             # Scoring criteria for all metrics
│
└── evaluation_results/
    ├── eval_*.json               # Individual results per agent + issue
    ├── 1_step_gptoss120b/        # GPT-OSS-120b, SR=1 (standard)
    ├── 1_step_qwen3_235b/        # Qwen3-235b, SR=1
    ├── 1_step_gpt4omini_8b/      # GPT-4o-mini-8b, SR=1
    ├── default_gptoss120b/       # GPT-OSS-120b, SR=5 (reduced)
    ├── default_qwen3_235b/       # Qwen3-235b, SR=5
    └── default_gpt4omini_8b/     # GPT-4o-mini-8b, SR=5
```

Each `evaluation_results/<run>/` directory contains:
- `consolidated_results.csv` — aggregated scores for all agents and issues
- `spearman_correlation_results.csv` — Spearman ρ per metric vs. gold standard
- Figures as `.png`, `.pdf`, `.pgf` and `.tex`

---

## Setup

```bash
cd mas-evaluation/metrics_evaluation
python3 -m venv venv
source venv/bin/activate
pip install pandas scipy scikit-learn matplotlib sentence-transformers
```

Set API key as environment variable:

```bash
export HELMHOLTZ_API_KEY=<key>
# or
export OPENAI_API_KEY=<key>
```

---

## Running the Evaluation

### Single Trajectory

```bash
python3 metrics_evaluation.py <trajectory_file> --agent OpenHands
python3 metrics_evaluation.py <file.traj> --agent SWE-Agent
python3 metrics_evaluation.py <file.txt> --agent MetaGPT --mas --global-plan
```

### Batch Processing (all issues)

```bash
# OpenHands
python3 batch_evaluation.py --agent OpenHands --logs-dir ../logs/openhands/logs

# SWE-Agent
python3 batch_evaluation.py --agent SWE-Agent --logs-dir ../logs/swe-agent

# MetaGPT (Multi-Agent + Global Plan)
python3 batch_evaluation.py --agent MetaGPT --logs-dir ../logs/metagpt --mas --global-plan

# live-swe-agent
python3 batch_evaluation.py --agent live-swe-agent --logs-dir ../logs/live-swe-agent

# With custom sampling rate
python3 batch_evaluation.py --agent SWE-Agent --logs-dir ../logs/swe-agent --sample-rate 1
```

### Consolidate Results

```bash
python3 consolidate_results.py
```

Reads all `eval_*.json` in the current directory and writes `consolidated_results.csv`.

### Display Trajectory

```bash
python3 review_trace.py --agent OpenHands --task scikit-learn__scikit-learn-12585
```

---

## Analysis and Validation

### Descriptive Analysis (RQ1, RQ3)

```bash
source venv/bin/activate
python3 descriptive_analysis.py
```

### Visualize Manual Annotations (RQ2)

```bash
python3 annotation_analysis.py
```

### Spearman Correlation Auto vs. Human (RQ2)

```bash
python3 spearman_correlation.py
```

Reads `evaluation_results/consolidated_results.csv` and `manual_annotations.csv`.
Writes `spearman_correlation_results.csv` to the current results directory.

### Weighted Kappa: SR Stability + Cross-Model (RQ4.2, RQ5)

```bash
python3 kappa_sample_rate_comparison.py
```

### AUC Predictor Analysis (RQ4.3)

```bash
python3 auc_predictor_analysis.py
```

Reads from `evaluation_results/1_step_gptoss120b/consolidated_results.csv`.

### Spearman SR=1 vs. SR=5 (RQ4.2)

```bash
source venv/bin/activate
python3 plot_spearman_sr_rq4.py
```

---

## Figures

All scripts save figures as `.png`, `.pdf`, `.pgf` and `.tex`.
The `.tex` file contains `\includegraphics[width=\linewidth]{name.pdf}` and can be
included in LaTeX directly via `\input{name.tex}`.

### Figures per Script

| Script | Generated Figures |
|--------|-------------------|
| `descriptive_analysis.py` | `cost_benefit_scatter`, `radar_chart_agents`, `boxplots_per_metric`, `boxplots_per_metric_4x2`, `agent_comparison_heatmap`, `mas_metrics_detail`, `mas_metrics_detail_2x2`, `success_vs_failure_comparison`, `metric_correlation_matrix` |
| `spearman_correlation.py` | `correlation_scatterplots`, `correlation_barchart` |
| `annotation_analysis.py` | `annotation_distribution`, `annotation_distribution_by_agent`, `annotation_heatmap_by_agent`, `annotation_spread_heatmap`, `interrater_agreement` |
| `plot_spearman_sr_rq4.py` | `spearman_sr_rq4` |
| `auc_predictor_analysis.py` | `auc_predictor_plot` |
| `kappa_sample_rate_comparison.py` | `kappa_sample_rate_comparison`, `spearman_sample_rate_comparison`, `kappa_cross_model_default_(sr=5)`, `kappa_cross_model_1_step_(sr=1)` |

---

## TikZ Figures (Chapters 2–4)

The conceptual figures of the thesis are available as TikZ source code in `../figures/`.

```bash
cd ../figures

./build_figure.sh fig_blackbox_vs_glassbox
./build_figure.sh fig_agent_architecture
./build_figure.sh fig_agent_evaluation
./build_figure.sh fig_eval_pipeline
./build_figure.sh fig_stratified_sample
./build_figure.sh fig_pipeline_flow
./build_figure.sh fig_prompt_anatomy
```

`build_figure.sh` generates a `.png` (300 dpi) and a `.pgf` file per figure.

| File | Content |
|------|---------|
| `fig_blackbox_vs_glassbox` | Black-Box vs. Glass-Box comparison, symmetric layout with divider in the center |
| `fig_agent_architecture` | LLM agent architecture, 4 columns (Wang 2024) |
| `fig_agent_evaluation` | Evaluation dimensions (Siegmund 2025), 7 questions with straight arrows |
| `fig_eval_pipeline` | 3-column pipeline (Execution → Transformation → Analysis), horizontal arrows with `\|-` trick |
| `fig_stratified_sample` | Pyramid Easy/Medium/Hard, Hard label inside, N=15 tasks |
| `fig_pipeline_flow` | Flowchart with 6 steps and TikZ icons |
| `fig_prompt_anatomy` | Sandwich block diagram system prompt, all boxes equally shaded, text aligned |

---

## CLI Arguments (batch_evaluation.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | required | `OpenHands`, `SWE-Agent`, `MetaGPT`, `live-swe-agent` |
| `--logs-dir` | required | Directory containing trajectory files |
| `--mas` | `false` | Enable multi-agent system mode |
| `--global-plan` | `false` | Enable metric M2.3 (only useful for MetaGPT) |
| `--sample-rate` | `5` | Sampling rate for M3.1 and M3.2 |
| `--context-window-steps` | `8` | Window size for M4.1 |
| `--context-sample-rate` | `4` | Sampling rate for M4.1 |
| `--start-from` | `1` | Start from issue N |
| `--only-issue` | – | Evaluate only one specific issue |
| `--prompts-file` | – | Path to alternative prompts file |
