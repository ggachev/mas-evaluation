#!/usr/bin/env python3
"""
Consolidate evaluation results from all eval_*.json files into a single CSV.
Extracts only the overall scores for each metric.
"""

import json
import csv
import os
from pathlib import Path
from typing import Any, Optional


def extract_score(metric_data: Any, key: str = "score") -> Optional[float]:
    """Extract a score from metric data, handling various cases."""
    if metric_data is None:
        return None
    if isinstance(metric_data, str):
        # Handle "N/A - Single Agent" cases
        return None
    if isinstance(metric_data, dict):
        return metric_data.get(key)
    return None


def process_eval_file(filepath: Path) -> dict:
    """Process a single evaluation JSON file and extract all scores."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get("meta", {})
    agent_name = meta.get("agent", "Unknown")
    task_id = meta.get("task", "Unknown")

    # Extract scores for each metric
    result = {
        "agent_name": agent_name,
        "issue_name": task_id,
    }

    # M1.1 - Task Success Rate (boolean → 1/0)
    m1_1 = data.get("metric_1_1_task_success_rate", {})
    if isinstance(m1_1, dict):
        success = m1_1.get("success")
        result["M1.1"] = 1 if success else (0 if success is False else None)
    else:
        result["M1.1"] = None

    # M1.2 - Resource Efficiency (multiple values - we'll use total_cost_usd as primary)
    m1_2 = data.get("metric_1_2_resource_efficiency", {})
    if isinstance(m1_2, dict):
        result["M1.2_cost_usd"] = m1_2.get("total_cost_usd")
        result["M1.2_tokens"] = m1_2.get("total_tokens")
        result["M1.2_duration_s"] = m1_2.get("duration_seconds")
        result["M1.2_steps"] = m1_2.get("step_count")
    else:
        result["M1.2_cost_usd"] = None
        result["M1.2_tokens"] = None
        result["M1.2_duration_s"] = None
        result["M1.2_steps"] = None

    # M2.1 - Loop Detection (boolean → 1/0)
    m2_1 = data.get("metric_2_1_loop_detection", {})
    if isinstance(m2_1, dict):
        loop_detected = m2_1.get("loop_detected")
        result["M2.1"] = 1 if loop_detected else (0 if loop_detected is False else None)
    else:
        result["M2.1"] = None

    # M2.2 - Trajectory Efficiency
    result["M2.2"] = extract_score(data.get("metric_2_2_trajectory_efficiency"))

    # M2.3 - Global Strategy Consistency
    result["M2.3"] = extract_score(data.get("metric_2_3_global_strategy_consistency"))

    # M2.4 - Stepwise Reasoning Quality (overall_score)
    result["M2.4"] = extract_score(data.get("metric_2_4_stepwise_reasoning_quality"), "overall_score")

    # M2.5 - Role Adherence
    result["M2.5"] = extract_score(data.get("metric_2_5_role_adherence"))

    # M3.1 - Tool Selection Quality (overall_score)
    result["M3.1"] = extract_score(data.get("metric_3_1_tool_selection_quality"), "overall_score")

    # M3.2 - Tool Execution Success (success_rate)
    result["M3.2"] = extract_score(data.get("metric_3_2_tool_execution_success"), "success_rate")

    # M3.3 - Tool Usage Efficiency (efficiency_score)
    result["M3.3"] = extract_score(data.get("metric_3_3_tool_usage_efficiency"), "efficiency_score")

    # M4.1 - Context Utilization & Consistency (overall_score)
    result["M4.1"] = extract_score(data.get("metric_4_1_context_utilization_consistency"), "overall_score")

    # M5.1 - Communication Efficiency (MAS only)
    result["M5.1"] = extract_score(data.get("metric_5_1_communication_efficiency"))

    # M5.2 - Information Diversity (MAS only)
    result["M5.2"] = extract_score(data.get("metric_5_2_information_diversity"))

    # M5.3 - Path Redundancy (MAS only)
    result["M5.3"] = extract_score(data.get("metric_5_3_unique_path_redundancy"))

    # M5.4 - Agent Invocation Distribution (MAS only)
    result["M5.4"] = extract_score(data.get("metric_5_4_agent_invocation_distribution"))

    return result


def main():
    # Find all eval_*.json files
    script_dir = Path(__file__).parent
    results_dir = script_dir / "evaluation_results"

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return

    eval_files = sorted(results_dir.glob("eval_*.json"))

    if not eval_files:
        print(f"No eval_*.json files found in {results_dir}")
        return

    print(f"Found {len(eval_files)} evaluation files")

    # Process all files
    all_results = []
    for filepath in eval_files:
        try:
            result = process_eval_file(filepath)
            all_results.append(result)
            print(f"  Processed: {filepath.name}")
        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    # Define CSV columns
    columns = [
        "agent_name", "issue_name",
        "M1.1", "M1.2_cost_usd", "M1.2_tokens", "M1.2_duration_s", "M1.2_steps",
        "M2.1", "M2.2", "M2.3", "M2.4", "M2.5",
        "M3.1", "M3.2", "M3.3",
        "M4.1",
        "M5.1", "M5.2", "M5.3", "M5.4"
    ]

    # Write CSV
    output_file = results_dir / "consolidated_results.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nCSV written to: {output_file}")
    print(f"Total rows: {len(all_results)}")


if __name__ == "__main__":
    main()
