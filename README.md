# Evaluation of Autonomous Coding Agents

This repository contains the complete evaluation pipeline for my master's thesis at the University of Leipzig.
It evaluates agent trajectories from four systems (OpenHands, SWE-agent, Live-SWE-agent, MetaGPT) using deterministic metrics and LLM-as-a-Judge.

Detailed documentation of the analysis scripts and figures:
→ [`metrics_evaluation/README.md`](metrics_evaluation/README.md)

---

## Directory Structure

```
mas-evaluation/
├── logs/                        # Raw agent trajectories (log files)
│   ├── openhands/
│   ├── swe-agent/
│   ├── live-swe-agent/
│   └── metagpt/
│
├── diffs/                       # Generated patches per agent and issue
│   ├── openhands/
│   ├── swe-agent/
│   ├── live-swe-agent/
│   └── metagpt/
│
├── metrics_evaluation/          # Evaluation pipeline (main module)
│   ├── metrics_evaluation.py    # Main evaluation script
│   ├── batch_evaluation.py      # Batch processing of all issues
│   ├── consolidate_results.py   # Consolidates eval_*.json into consolidated_results.csv
│   ├── agent_parsers/           # Parsers for the various log formats
│   ├── descriptive_analysis.py  # Descriptive analysis + plots (RQ1, RQ3)
│   ├── annotation_analysis.py   # Visualization of manual annotations (RQ2)
│   ├── spearman_correlation.py  # Spearman correlation auto vs. human (RQ2)
│   ├── kappa_sample_rate_comparison.py  # Weighted Kappa SR + Cross-Model (RQ4, RQ5)
│   ├── auc_predictor_analysis.py        # AUC predictor analysis (RQ4.3)
│   ├── plot_spearman_sr_rq4.py          # Spearman SR=1 vs. SR=5 (RQ4.2)
│   ├── manual_annotations.csv   # Manual expert ratings (Rater 1)
│   ├── evaluation_split_final.txt       # The 15 selected SWE-bench Verified issues
│   ├── evaluation_results/      # Results, CSVs and figures
│   │   ├── 1_step_gptoss120b/   # GPT-OSS-120b, SR=1 (standard configuration)
│   │   ├── 1_step_qwen3_235b/   # Qwen3-235b, SR=1
│   │   ├── 1_step_gpt4omini_8b/ # GPT-4o-mini-8b, SR=1
│   │   ├── default_gptoss120b/  # GPT-OSS-120b, SR=5 (reduced configuration)
│   │   ├── default_qwen3_235b/  # Qwen3-235b, SR=5
│   │   └── default_gpt4omini_8b/# GPT-4o-mini-8b, SR=5
│   └── README.md                # Detailed documentation
│
├── README.md                    # Detailed project documentation
├── run_all_evaluations.sh       # Run all evaluations via batch
└── run_all_metrics.sh           # Evaluate specific traces selectively
```

---

## Setup

```bash
cd mas-evaluation/metrics_evaluation
python3 -m venv venv
source venv/bin/activate
pip install openai pandas scipy scikit-learn matplotlib sentence-transformers

export HELMHOLTZ_API_KEY="<key>"
# or
export OPENAI_API_KEY="<key>"
```

---

## Running the Evaluation

```bash
cd mas-evaluation

# Single trajectory
python metrics_evaluation/metrics_evaluation.py <trajectory_file> --agent OpenHands

# Batch processing
python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs
python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent
python metrics_evaluation/batch_evaluation.py --agent live-swe-agent --logs-dir logs/live-swe-agent
python metrics_evaluation/batch_evaluation.py --agent MetaGPT --logs-dir logs/metagpt --mas --global-plan

# Consolidate results
cd metrics_evaluation
python consolidate_results.py
```

---

## Metrics

### Category 1: Results and Costs

| Metric | Type | Description |
|--------|------|-------------|
| M1.1 Task Success Rate | Manual | Binary success from manual labels |
| M1.2 Resource Efficiency | Deterministic | Costs, tokens, duration, step count |

### Category 2: Strategy and Navigation

| Metric | Type | Description |
|--------|------|-------------|
| M2.1 Loop Detection | Deterministic | Hash-based detection of repeated sequences |
| M2.2 Trajectory Efficiency | LLM-Judge | Efficiency of the solution path |
| M2.3 Global Strategy Consistency | LLM-Judge | Plan formulation and adherence (MetaGPT only) |
| M2.4 Stepwise Reasoning Quality | LLM-Judge | Logical quality per step |
| M2.5 Role Adherence | LLM-Judge | Adherence to the agent role |

### Category 3: Tools

| Metric | Type | Description |
|--------|------|-------------|
| M3.1 Tool Selection Quality | LLM-Judge | Appropriateness of tool choice |
| M3.2 Tool Execution Success | LLM-Judge | Technical execution rate |
| M3.3 Tool Usage Efficiency | Deterministic | Context pollution measurement |

### Category 4: Knowledge and Context

| Metric | Type | Description |
|--------|------|-------------|
| M4.1 Context Utilization | LLM-Judge | Consistency in sliding window |

### Category 5: Multi-Agent Systems (MetaGPT only)

| Metric | Type | Description |
|--------|------|-------------|
| M5.1 Communication Efficiency | LLM-Judge | Signal-to-noise ratio of communication |
| M5.2 Information Diversity | Embeddings | Diversity of agent messages |
| M5.3 Path Redundancy | Deterministic | Ping-pong pattern detection |
| M5.4 Agent Invocation Distribution | Deterministic | Workload distribution (Shannon entropy) |

---

## Output Format

Results are stored in `evaluation_results/`:

```json
{
  "meta": {
    "agent": "SWE-Agent",
    "task": "scikit-learn__scikit-learn-12585",
    "timestamp": "2025-12-28 17:23:44",
    "is_multi_agent_system": false,
    "llm_judge_model": "GPT-OSS-120b"
  },
  "metric_1_1_task_success_rate": {"success": true, "source": "manual_labels"},
  "metric_1_2_resource_efficiency": {"total_cost_usd": 0.017, "total_tokens": 68524},
  "metric_2_2_trajectory_efficiency": {"score": 0.85, "reasoning": "..."},
  "metric_5_1_communication_efficiency": "N/A - Single Agent"
}
```

---

## Supported Agent Formats

| Agent | Format | Notes |
|-------|--------|-------|
| OpenHands | JSON (`history` array) | Costs from `metrics.accumulated_cost` |
| SWE-Agent | `.traj` (JSON) + `.config.yaml` | Task from `problem_statement.text` |
| Live-SWE-Agent | `.traj` + `.config.yaml` | Similar to SWE-Agent |
| MetaGPT | `.txt` / `.log` | Multi-agent detection from `AgentName(Role)` patterns |

---

## LLM-Judge Configuration

```python
BASE_URL_JUDGE = "https://api.helmholtz-blablador.fz-juelich.de/v1"
MODEL_JUDGE    = "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
MODEL_EMBEDDING = "text-embedding-3-small"
CONTEXT_WINDOW_SIZE = 131000
```

```python
BASE_URL_JUDGE = "https://api.helmholtz-blablador.fz-juelich.de/v1"
MODEL_JUDGE    = "2 - Qwen3 235, a great model from Alibaba with a long context size"
MODEL_EMBEDDING = "text-embedding-3-small"
CONTEXT_WINDOW_SIZE = 204000
```

```python
BASE_URL_JUDGE = "http://91.99.56.205:4000/v1"
MODEL_JUDGE    = "gpt-4o-mini"
MODEL_EMBEDDING = "text-embedding-3-small"
CONTEXT_WINDOW_SIZE = 128000
```
