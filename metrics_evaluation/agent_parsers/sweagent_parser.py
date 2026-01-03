import json
import os
import glob
import re
from typing import List, Dict, Any

# Eigene Module importieren (Lokal, da Script im gleichen Ordner ausgeführt wird)
from evaluation_data_models import EvaluationTrace, StandardStep

def parse_sweagent_trajectory(file_path: str, agent_name_override: str) -> EvaluationTrace:
    # 1. Trajectory File (.traj) laden
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps = []
    trajectory = data.get("trajectory", [])

    # Task ID aus Dateinamen (Fallback)
    filename = os.path.basename(file_path)
    task_id = filename.replace(".traj", "")

    # 2. Try to load task description from .config.yaml file in same directory
    task_description = None
    traj_dir = os.path.dirname(file_path)
    # Look for .config.yaml files (SWE-Agent naming convention)
    yaml_files = glob.glob(os.path.join(traj_dir, "*.config.yaml"))
    if not yaml_files:
        # Fallback to any .yaml file
        yaml_files = glob.glob(os.path.join(traj_dir, "*.yaml"))

    if yaml_files:
        try:
            # The YAML file contains a JSON string (possibly wrapped in quotes)
            with open(yaml_files[0], 'r', encoding='utf-8') as f:
                yaml_content = f.read()

            # Strip leading/trailing whitespace
            yaml_content = yaml_content.strip()

            # Strip leading/trailing quotes if present (YAML string format)
            # Two formats exist:
            # 1. Single-quoted YAML: '{"key": "value"}' - uses '' for escaped single quotes
            # 2. Double-quoted YAML: "{\"key\": \"value\"}" - uses \" for escaped double quotes

            if yaml_content.startswith("'"):
                # Single-quoted YAML format
                if "'" in yaml_content[1:]:
                    last_quote = yaml_content.rfind("'")
                    yaml_content = yaml_content[1:last_quote]
                else:
                    yaml_content = yaml_content[1:]
                # YAML uses '' to escape single quotes inside single-quoted strings
                yaml_content = yaml_content.replace("''", "'")
                # Handle literal control characters (actual newlines in the file)
                yaml_content = yaml_content.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n').replace('\t', '\\t')

            elif yaml_content.startswith('"'):
                # Double-quoted YAML format
                if '"' in yaml_content[1:]:
                    last_quote = yaml_content.rfind('"')
                    yaml_content = yaml_content[1:last_quote]
                else:
                    yaml_content = yaml_content[1:]

                # Handle YAML line continuations: backslash + newline + optional whitespace
                # This joins lines that were split for readability
                # Handle both \n and \r\n line endings
                yaml_content = re.sub(r'\\\r?\n\s*', '', yaml_content)

                # YAML double-quoted strings use \ followed by space for literal space
                # This often appears after line continuation: "...\\\n  \ to continue"
                yaml_content = yaml_content.replace('\\ ', ' ')

                # YAML double-quoted strings use \" for literal quotes
                # We need to unescape: \" -> "
                yaml_content = yaml_content.replace('\\"', '"')
                # Also handle escaped backslashes: \\ -> \
                yaml_content = yaml_content.replace('\\\\', '\\')
                # Handle escaped newlines that are literal \n in the file (not actual newlines)
                # These are already in the correct format for JSON

            # Parse as JSON
            config_data = json.loads(yaml_content)

            # Extract problem statement
            problem_statement = config_data.get("problem_statement", {})
            if isinstance(problem_statement, dict):
                task_description = problem_statement.get("text", "")
                # Use problem_statement.id as task_id if available
                ps_id = problem_statement.get("id")
                if ps_id:
                    task_id = ps_id
        except Exception as e:
            print(f"Warning: Could not parse YAML config for {file_path}: {e}")

    # Add the task description as the first "user" step if found
    if task_description:
        steps.append(StandardStep(
            step_id=0,
            role="user",
            type="instruction",
            content=task_description,
            metadata={},
            agent_name="user"
        ))
    
    total_duration = 0.0

    # 3. Iteration durch Schritte
    for i, item in enumerate(trajectory):
        # SWE-agent hat oft thought -> action -> observation im gleichen Block
        
        # Thought
        thought_content = item.get("thought", "")
        if thought_content:
            steps.append(StandardStep(
                step_id=len(steps),
                role="agent",
                type="thought",
                content=thought_content,
                metadata={},
                agent_name=agent_name_override
            ))

        # Action
        action_content = item.get("action", "")
        if action_content:
            steps.append(StandardStep(
                step_id=len(steps),
                role="agent",
                type="action",
                content=action_content,
                metadata={"execution_time": item.get("execution_time")},
                agent_name=agent_name_override
            ))
        
        # Observation
        obs_content = item.get("observation", "")
        if obs_content:
            steps.append(StandardStep(
                step_id=len(steps),
                role="tool",
                type="observation",
                content=str(obs_content), # Sicherstellen dass es String ist
                metadata={},
                agent_name=agent_name_override
            ))

        # Time aggregieren
        total_duration += item.get("execution_time", 0.0)

    # 4. Extract cost/tokens from info.model_stats in the .traj file
    total_cost = 0.0
    total_tokens = 0

    info = data.get("info", {})
    model_stats = info.get("model_stats", {})
    if model_stats:
        total_cost = model_stats.get("instance_cost", 0.0) or 0.0
        tokens_sent = model_stats.get("tokens_sent", 0) or 0
        tokens_received = model_stats.get("tokens_received", 0) or 0
        total_tokens = tokens_sent + tokens_received

    return EvaluationTrace(
        agent_name=agent_name_override,
        task_id=task_id,
        steps=steps,
        total_cost=total_cost,
        total_tokens=total_tokens,
        duration_seconds=total_duration,
        success_status=None, # Wird später über Labels gematched
        is_multi_agent=False # SWE-agent ist single agent
    )
