#!/usr/bin/env python3
"""
Batch Evaluation Script
Runs metrics_evaluation.py for all trajectory files in a logs directory.

Usage (from mas-evaluation folder):
    python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs
    python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent/logs

All parameters from metrics_evaluation.py are supported.
"""

import argparse
import os
import sys
import glob
import time
from datetime import datetime

# Add the metrics_evaluation directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from metrics_evaluation import main as run_evaluation


def find_trajectory_files(logs_dir: str, agent: str) -> list:
    """
    Find all trajectory files in issue_* folders or directly in logs_dir.
    - For OpenHands: looks for .json files
    - For SWE-Agent: looks for .traj files
    - For MetaGPT: looks for .txt or .log files
    - For live-swe-agent: looks for .traj.json or .json files
    Only looks for files directly in issue_N folders, not in subfolders.
    For MetaGPT/live-swe-agent, also looks for files directly in logs_dir if no issue_* folders found.
    """
    trajectory_files = []

    # Determine file extension based on agent type
    agent_lower = agent.lower()
    if agent_lower in ["swe-agent", "sweagent"]:
        extensions = [".traj"]
    elif agent_lower == "metagpt":
        extensions = [".txt", ".log"]
    elif agent_lower in ["live-swe-agent", "live_swe_agent", "livesweagent", "mini-swe-agent"]:
        extensions = [".traj.json", ".json"]
    else:
        extensions = [".json"]

    # Find all issue_* directories
    issue_pattern = os.path.join(logs_dir, "issue_*")
    issue_dirs = sorted(glob.glob(issue_pattern), key=lambda x: int(x.split("issue_")[-1]) if x.split("issue_")[-1].isdigit() else 0)

    for issue_dir in issue_dirs:
        if not os.path.isdir(issue_dir):
            continue

        # Find trajectory files directly in the issue folder (not in subfolders)
        traj_files = [f for f in os.listdir(issue_dir)
                      if any(f.endswith(ext) for ext in extensions) and not f.startswith(".")]

        if len(traj_files) == 1:
            trajectory_files.append(os.path.join(issue_dir, traj_files[0]))
        elif len(traj_files) > 1:
            print(f"Warning: Multiple trajectory files in {issue_dir}, using first: {traj_files[0]}")
            trajectory_files.append(os.path.join(issue_dir, traj_files[0]))
        else:
            print(f"Warning: No trajectory file found in {issue_dir} (looking for {extensions})")

    # For MetaGPT/live-swe-agent: if no issue_* folders found, look for files directly in logs_dir
    if not trajectory_files and agent_lower in ["metagpt", "live-swe-agent", "live_swe_agent", "livesweagent", "mini-swe-agent"]:
        direct_files = [f for f in os.listdir(logs_dir)
                        if any(f.endswith(ext) for ext in extensions)
                        and not f.startswith(".")
                        and os.path.isfile(os.path.join(logs_dir, f))]
        trajectory_files = sorted([os.path.join(logs_dir, f) for f in direct_files])

    return trajectory_files


def run_batch_evaluation(
    logs_dir: str,
    agent: str,
    mas: bool = False,
    sample_rate: int = 5,
    context_window_steps: int = 8,
    context_sample_rate: int = 4,
    start_from: int = 1,
    only_issue: int = None
):
    """
    Run evaluation for all trajectory files in the logs directory.
    """
    print(f"\n{'='*60}")
    print(f"BATCH EVALUATION")
    print(f"{'='*60}")
    print(f"Logs directory: {logs_dir}")
    print(f"Agent: {agent}")
    print(f"MAS mode: {mas}")
    print(f"Sample rate (3.1/3.2): {sample_rate}")
    print(f"Context window (4.1): {context_window_steps} steps")
    print(f"Context sample rate (4.1): every {context_sample_rate}th step")
    print(f"{'='*60}\n")

    # Find all trajectory files
    trajectory_files = find_trajectory_files(logs_dir, agent)

    if not trajectory_files:
        print(f"Error: No trajectory files found in {logs_dir}")
        return

    print(f"Found {len(trajectory_files)} trajectory file(s):\n")
    for i, f in enumerate(trajectory_files, 1):
        issue_num = os.path.basename(os.path.dirname(f))
        filename = os.path.basename(f)
        print(f"  {i:2}. [{issue_num}] {filename}")
    print()

    # Filter based on start_from or only_issue
    if only_issue:
        trajectory_files = [f for f in trajectory_files
                           if f"issue_{only_issue}" in f]
        if not trajectory_files:
            print(f"Error: No trajectory found for issue_{only_issue}")
            return
    elif start_from > 1:
        trajectory_files = [f for f in trajectory_files
                           if int(os.path.basename(os.path.dirname(f)).split("_")[-1]) >= start_from]

    # Run evaluation for each file
    total_files = len(trajectory_files)
    successful = 0
    failed = []
    start_time = time.time()

    for idx, file_path in enumerate(trajectory_files, 1):
        issue_folder = os.path.basename(os.path.dirname(file_path))
        filename = os.path.basename(file_path)

        print(f"\n{'='*60}")
        print(f"[{idx}/{total_files}] Processing {issue_folder}: {filename}")
        print(f"{'='*60}")

        try:
            run_evaluation(
                file_path=file_path,
                agent_type_arg=agent,
                force_mas_arg=mas,
                sample_rate=sample_rate,
                context_window_steps=context_window_steps,
                context_sample_rate=context_sample_rate
            )
            successful += 1
            print(f"\n✓ Completed {issue_folder}")
        except Exception as e:
            failed.append((issue_folder, str(e)))
            print(f"\n✗ Failed {issue_folder}: {e}")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total_files} | Successful: {successful} | Failed: {len(failed)}")
    print(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    if failed:
        print(f"\nFailed evaluations:")
        for issue, error in failed:
            print(f"  - {issue}: {error}")

    print(f"\nResults saved to: metrics_evaluation/evaluation_results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch evaluation for multiple trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all OpenHands trajectories
  python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs

  # Evaluate all SWE-Agent trajectories with custom sample rate
  python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent/logs --sample-rate 3

  # Evaluate MetaGPT trajectories (multi-agent system)
  python metrics_evaluation/batch_evaluation.py --agent MetaGPT --logs-dir logs/metagpt --mas

  # Evaluate live-swe-agent trajectories
  python metrics_evaluation/batch_evaluation.py --agent live-swe-agent --logs-dir logs/live-swe-agent

  # Start from issue 5 (skip 1-4)
  python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs --start-from 5

  # Evaluate only issue 10
  python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs --only-issue 10
        """
    )

    # Required arguments
    parser.add_argument("--logs-dir", required=True,
                        help="Directory containing issue_* folders with trajectory JSON files")
    parser.add_argument("--agent", required=True,
                        help="Agent type (e.g., OpenHands, SWE-Agent, MetaGPT, live-swe-agent)")

    # Optional arguments (matching metrics_evaluation.py)
    parser.add_argument("--mas", action="store_true",
                        help="Force MAS (Multi-Agent System) evaluation")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Sample rate for metrics 3.1 and 3.2 (default: 5)")
    parser.add_argument("--context-window-steps", type=int, default=8,
                        help="Sliding window size for metric 4.1 (default: 8)")
    parser.add_argument("--context-sample-rate", type=int, default=4,
                        help="Sample rate for metric 4.1 (default: 4)")

    # Batch-specific arguments
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from issue N (skip earlier issues)")
    parser.add_argument("--only-issue", type=int, default=None,
                        help="Evaluate only a specific issue number")

    args = parser.parse_args()

    run_batch_evaluation(
        logs_dir=args.logs_dir,
        agent=args.agent,
        mas=args.mas,
        sample_rate=args.sample_rate,
        context_window_steps=args.context_window_steps,
        context_sample_rate=args.context_sample_rate,
        start_from=args.start_from,
        only_issue=args.only_issue
    )
