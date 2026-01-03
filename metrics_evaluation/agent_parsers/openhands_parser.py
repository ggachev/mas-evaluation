import json
import os
import datetime
from typing import List, Dict, Any
# from collections import defaultdict, Counter # Nicht direkt hier benötigt

# Eigene Module importieren
from evaluation_data_models import EvaluationTrace, StandardStep

# Maximum gap between steps to count as active work time (in seconds)
# Gaps larger than this are assumed to be idle/paused time
MAX_GAP_SECONDS = 300  # 5 minutes

def parse_openhands_trajectory(file_path: str, agent_name_override: str) -> EvaluationTrace:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps = []
    max_cost = 0.0
    max_tokens = 0
    timestamps = []  # Collect all valid timestamps for duration calculation
    agents_seen = set()

    filename = os.path.basename(file_path)
    task_id = filename.replace(".json", "")

    for entry in data:
        ts = entry.get("timestamp")
        if ts:
            try:
                t = datetime.datetime.fromisoformat(ts)
                timestamps.append(t)
            except: pass

        m = entry.get("metrics", {}) or entry.get("llm_metrics", {})
        if m and isinstance(m, dict):
            c = m.get("accumulated_cost")
            if isinstance(c, (int, float)) and c > max_cost: max_cost = c

            t_acc = m.get("accumulated_token_usage")
            # accumulated_token_usage can be a dict with prompt_tokens and completion_tokens
            if isinstance(t_acc, dict):
                prompt_t = t_acc.get("prompt_tokens", 0) or 0
                completion_t = t_acc.get("completion_tokens", 0) or 0
                total_acc = prompt_t + completion_t
                if total_acc > max_tokens:
                    max_tokens = total_acc
            elif isinstance(t_acc, (int, float)) and t_acc > max_tokens:
                max_tokens = int(t_acc)

        source = entry.get("source", "")
        agent_id = "default"
        if source == "agent":
            agent_id = entry.get("extras", {}).get("agent_type", "OpenHandsAgent")
            agents_seen.add(agent_id)

        role, typ, content, meta = "unknown", "unknown", "", {"raw_id": entry.get("id")}
        action_type = entry.get("action", "")
        observation_type = entry.get("observation", "")

        if source == "agent":
            role = "agent"
            # Check if this is an observation (result of a previous action)
            if observation_type:
                typ = "observation"
                content = entry.get("content", "")
                meta["exit_code"] = entry.get("extras", {}).get("metadata", {}).get("exit_code")
                meta["observation_type"] = observation_type
            # Otherwise it's an action
            elif action_type == "think":
                typ, content = "thought", entry.get("args", {}).get("thought", "")
            elif action_type == "message":
                typ, content = "thought", entry.get("args", {}).get("content", "")
            elif action_type in ["run", "run_ipython", "browse", "write", "read", "edit"]:
                # Real tool actions
                typ = "action"
                content = entry.get("args", {}).get("command", "") or entry.get("args", {}).get("path", "") or json.dumps(entry.get("args", {}))
                meta["action_type"] = action_type
                if action_type == "run":
                    meta["tool"] = "execute_bash"
            elif action_type in ["finish", "reject", "delegate"]:
                # Terminal/control actions - treat as thought/decision
                typ, content = "thought", entry.get("message", "") or json.dumps(entry.get("args", {}))
                meta["action_type"] = action_type
            elif action_type:
                # Other actions (system, agent_state, etc.) - skip or treat as thought
                typ, content = "thought", entry.get("message", "") or ""
                meta["action_type"] = action_type
        elif source == "tool":
            role, typ = "tool", "observation"
            content = entry.get("content", "")
            meta["exit_code"] = entry.get("extras", {}).get("metadata", {}).get("exit_code")
        elif source == "user":
            role, typ, content = "user", "instruction", entry.get("args", {}).get("content", "")

        if content:
            steps.append(StandardStep(len(steps), role, typ, content, meta, agent_id))

    # Calculate duration with capped gaps
    # Sum time between consecutive timestamps, but cap each gap at MAX_GAP_SECONDS
    # This excludes idle time when agent was paused/stopped
    duration = 0.0
    if len(timestamps) > 1:
        timestamps.sort()  # Ensure chronological order
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            # Cap the gap to exclude idle/paused time
            duration += min(gap, MAX_GAP_SECONDS)

    return EvaluationTrace(
        agent_name_override, task_id, steps, max_cost, max_tokens, duration, None,
        is_multi_agent=(len(agents_seen) > 1)
    )