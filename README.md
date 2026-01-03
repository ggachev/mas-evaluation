# Metrics Evaluation Framework for Autonomous Coding Agents

A comprehensive evaluation framework for measuring the performance of autonomous coding agents (OpenHands, SWE-Agent, Live-SWE-Agent, MetaGPT). This system evaluates agent trajectories using both **deterministic metrics** and **LLM-as-a-Judge** probabilistic metrics.

## Overview

This project is part of a Master's thesis on evaluating multi-agent coding systems. It analyzes agent execution logs (trajectories) to measure:

- Task success and resource efficiency
- Strategic planning and navigation quality
- Tool usage patterns and effectiveness
- Context utilization and memory consistency
- Multi-agent communication and coordination (for MAS)

## Installation

### Requirements

- Python 3.8+
- API access to an LLM judge (Helmholtz Blablador or OpenAI)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd mas-evaluation

# Install dependencies
pip install openai numpy sentence-transformers

# Set API key
export HELMHOLTZ_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

## Usage

```bash
cd mas-evaluation

# Single trajectory
python metrics_evaluation/metrics_evaluation.py <trajectory_file> --agent <AgentType>

# Batch evaluation (all issues)
python metrics_evaluation/batch_evaluation.py --agent <AgentType> --logs-dir <logs_directory>
```

### Advanced Options

```bash
# Custom sample rates
python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands \
    --sample-rate 3 --context-window-steps 10

# Start from specific issue
python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent --start-from 5

# Evaluate single issue only
python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent --only-issue 10
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | required | Agent type: `OpenHands`, `SWE-Agent`, `Live-SWE-Agent`, `MetaGPT` |
| `--mas` | false | Force multi-agent system evaluation |
| `--sample-rate` | 5 | Sample rate for tool metrics (3.1, 3.2) |
| `--context-window-steps` | 8 | Sliding window size for context metric (4.1) |
| `--context-sample-rate` | 4 | Sample rate for context windows |
| `--start-from` | 1 | Start from issue N (batch mode) |

## Metrics

### Category 1: Results & Costs

| Metric | Type | Description |
|--------|------|-------------|
| 1.1 Task Success Rate | Manual Labels | Binary success from labeled data |
| 1.2 Resource Efficiency | Deterministic | Cost, tokens, duration, step count |

### Category 2: Strategy & Navigation

| Metric | Type | Description |
|--------|------|-------------|
| 2.1 Loop Detection | Deterministic | Hash-based repeated sequence detection |
| 2.2 Trajectory Efficiency | LLM Judge | Path efficiency (not success) |
| 2.3 Global Strategy Consistency | LLM Judge | Plan formulation and adherence |
| 2.4 Stepwise Reasoning Quality | LLM Judge | Per-step logical flow analysis |
| 2.5 Role Adherence | LLM Judge | Behavioral compliance evaluation |

### Category 3: Tools (Actions)

| Metric | Type | Description |
|--------|------|-------------|
| 3.1 Tool Selection Quality | LLM Judge | Appropriateness of tool choices |
| 3.2 Tool Execution Success | LLM Judge | Technical execution success rate |
| 3.3 Tool Usage Efficiency | Deterministic | Context pollution measurement |

### Category 4: Knowledge (Memory)

| Metric | Type | Description |
|--------|------|-------------|
| 4.1 Context Utilization | LLM Judge | Consistency within sliding windows |

### Category 5: Multi-Agent Systems

| Metric | Type | Description |
|--------|------|-------------|
| 5.1 Communication Efficiency | LLM Judge | Signal-to-noise in agent communication |
| 5.2 Information Diversity | Embeddings | Diversity of agent messages |
| 5.3 Path Redundancy | Deterministic | Ping-pong pattern detection |
| 5.4 Agent Invocation Distribution | Deterministic | Work distribution (Shannon entropy) |

## Output

Results are saved to `evaluation_results/`:

- `eval_{agent}_{task_id}.json` - Evaluation results
- `api_logs_{agent}_{task_id}.json` - LLM API call logs

### Example Output Structure

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
  "metric_2_1_loop_detection": {"loop_detected": false},
  "metric_2_2_trajectory_efficiency": {"score": 0.85, "reasoning": "..."},
  "..."
}
```

## Project Structure

```
mas-evaluation/
├── agent_systems/               # Agent source code (reference implementations)
│   ├── chatdev/
│   ├── metagpt/
│   ├── mini-swe-agent/
│   ├── openhands/
│   ├── SWE-agent/
│   ├── SWE-bench/
│   └── SWE-search/
├── diffs/                       # Generated patches/diffs per agent
│   ├── live-swe-agent/
│   ├── metagpt/
│   ├── openhands/
│   └── swe-agent/
├── logs/                        # Agent execution trajectories
│   ├── live-swe-agent/
│   ├── metagpt/
│   ├── openhands/
│   └── swe-agent/
├── metrics_evaluation/          # Evaluation framework (this module)
│   ├── metrics_evaluation.py    # Main evaluation script
│   ├── batch_evaluation.py      # Batch processing script
│   ├── evaluation_prompts.py    # LLM prompts for judge metrics
│   ├── evaluation_data_models.py # Data models
│   ├── evaluation_labels_*.json # Manual success labels per agent
│   ├── agent_parsers/
│   │   ├── openhands_parser.py
│   │   ├── sweagent_parser.py
│   │   ├── live_sweagent_parser.py
│   │   └── metagpt_parser.py
│   ├── evaluation_results/      # Output directory
│   └── README.md
└── swe_bench_verified_issues/   # 15 SWE-bench verified issues for evaluation
```

## Supported Agent Formats

### OpenHands
- Format: JSON with `history` array
- Cost/tokens from `metrics.accumulated_cost`

### SWE-Agent
- Format: `.traj` (JSON) + `.config.yaml`
- Task from `problem_statement.text` in config

### Live-SWE-Agent
- Format: Similar to SWE-Agent
- Real-time execution logs

### MetaGPT
- Format: `.txt` or `.log` text files
- Multi-agent detection from `AgentName(Role)` patterns
- Supports multiple agents like Mike (Team Leader), Alex (Engineer), Alice (PM), Bob (Architect), David (DataAnalyst) and so on dynamically.

## Configuration

Default LLM configuration in `metrics_evaluation.py`:

```python
BASE_URL_JUDGE = "https://api.helmholtz-blablador.fz-juelich.de/v1"
MODEL_JUDGE = "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
MODEL_EMBEDDING = "text-embedding-3-small"
CONTEXT_WINDOW_SIZE = 131000
```

## License

Part of Master's thesis research.
