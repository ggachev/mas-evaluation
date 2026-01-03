import re
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple

from evaluation_data_models import EvaluationTrace, StandardStep

# Maximum gap between timestamps to count as active work time (in seconds)
# Gaps larger than this are assumed to be idle/paused time
MAX_GAP_SECONDS = 300  # 5 minutes


def parse_metagpt_trajectory(file_path: str, agent_name_override: str) -> EvaluationTrace:
    """
    Parse MetaGPT text-based log files into an EvaluationTrace.

    MetaGPT logs are multi-agent system logs with format:
    TIMESTAMP | LEVEL | module:function:line - message

    Key patterns:
    - Cost tracking: metagpt.utils.cost_manager:update_cost - Total running cost: $X.XXX ...
    - Agent reactions: metagpt.roles.di.role_zero:_react - AgentName(Role): ...
    - Commands: metagpt.roles.di.role_zero:_act:292 - Commands: [JSON]
    - Command outputs: metagpt.roles.di.role_zero:_act:294 - Commands outputs: ...
    - Messages: metagpt.environment.base_env:publish_message - publish_message: {JSON}
    - Observations: metagpt.roles.role:_observe - AgentName(Role) observed: [...]
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    steps = []
    timestamps = []
    agents_seen = set()
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    task_description = None

    # Extract task_id from filename
    import os
    filename = os.path.basename(file_path)
    task_id = filename.replace(".txt", "").replace(".log", "")

    # Regex patterns
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
    cost_pattern = re.compile(r'Total running cost: \$([0-9.]+).*?prompt_tokens: (\d+), completion_tokens: (\d+)')
    agent_react_pattern = re.compile(r'(\w+)\(([^)]+)\):\s*self\.rc\.state=')
    agent_observe_pattern = re.compile(r'(\w+)\(([^)]+)\) observed:')
    commands_start_pattern = re.compile(r'metagpt\.roles\.di\.role_zero:_act:\d+ - Commands:\s*$')
    commands_output_pattern = re.compile(r'metagpt\.roles\.di\.role_zero:_act:\d+ - Commands outputs:')
    publish_message_pattern = re.compile(r'publish_message: (\{.*\})')
    waiting_pattern = re.compile(r'(\w+)\(([^)]+)\): no news\. waiting\.')

    current_agent = "unknown"
    current_role = "unknown"
    i = 0

    while i < len(lines):
        line = lines[i]

        # Extract timestamp
        ts_match = timestamp_pattern.match(line)
        if ts_match:
            try:
                ts = datetime.datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(ts)
            except:
                pass

        # Extract cost and tokens (accumulative - take the last/max value)
        cost_match = cost_pattern.search(line)
        if cost_match:
            cost = float(cost_match.group(1))
            prompt_tokens = int(cost_match.group(2))
            completion_tokens = int(cost_match.group(3))
            if cost > total_cost:
                total_cost = cost
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        # Extract agent from react pattern
        react_match = agent_react_pattern.search(line)
        if react_match:
            current_agent = react_match.group(1)
            current_role = react_match.group(2)
            agents_seen.add(f"{current_agent}({current_role})")

        # Extract agent from observe pattern
        observe_match = agent_observe_pattern.search(line)
        if observe_match:
            agent_name = observe_match.group(1)
            role = observe_match.group(2)
            agents_seen.add(f"{agent_name}({role})")

            # Extract what was observed (rest of line after "observed:")
            obs_start = line.find("observed:")
            if obs_start != -1:
                observed_content = line[obs_start + 9:].strip()
                if observed_content:
                    steps.append(StandardStep(
                        step_id=len(steps),
                        role="agent",
                        type="observation",
                        content=f"{agent_name} observed: {observed_content}",
                        metadata={"agent_role": role},
                        agent_name=agent_name
                    ))

        # Extract waiting agents (just for agent tracking, not as steps)
        waiting_match = waiting_pattern.search(line)
        if waiting_match:
            agents_seen.add(f"{waiting_match.group(1)}({waiting_match.group(2)})")

        # Extract published messages
        publish_match = publish_message_pattern.search(line)
        if publish_match:
            try:
                msg_json = json.loads(publish_match.group(1))
                content = msg_json.get("content", "")
                sent_from = msg_json.get("sent_from", "")
                send_to = msg_json.get("send_to", [])
                role_type = msg_json.get("role", "")

                # Extract task description from first User message
                if task_description is None and "from User" in content:
                    task_description = content

                # Determine step type based on sender
                if sent_from == "" or "User" in content[:50]:
                    step_role = "user"
                    step_type = "instruction"
                else:
                    step_role = "agent"
                    step_type = "thought"  # Inter-agent communication
                    agents_seen.add(sent_from)

                if content:
                    steps.append(StandardStep(
                        step_id=len(steps),
                        role=step_role,
                        type=step_type,
                        content=content,
                        metadata={
                            "sent_from": sent_from,
                            "send_to": send_to,
                            "message_id": msg_json.get("id", "")
                        },
                        agent_name=sent_from if sent_from else "User"
                    ))
            except json.JSONDecodeError:
                pass

        # Extract Commands (actions)
        if commands_start_pattern.search(line):
            # Next line(s) contain the command JSON
            command_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Stop when we hit another log line with timestamp
                if timestamp_pattern.match(next_line):
                    break
                command_lines.append(next_line)
                i += 1

            command_content = '\n'.join(command_lines).strip()
            if command_content:
                # Try to parse as JSON to extract command details
                try:
                    commands = json.loads(command_content.replace("'", '"'))
                    for cmd in commands:
                        cmd_name = cmd.get("command_name", "unknown")
                        cmd_args = cmd.get("args", {})

                        steps.append(StandardStep(
                            step_id=len(steps),
                            role="agent",
                            type="action",
                            content=f"{cmd_name}: {json.dumps(cmd_args)}",
                            metadata={
                                "command_name": cmd_name,
                                "args": cmd_args
                            },
                            agent_name=current_agent
                        ))
                except:
                    # If JSON parsing fails, store raw content
                    if command_content and command_content != '[]':
                        steps.append(StandardStep(
                            step_id=len(steps),
                            role="agent",
                            type="action",
                            content=command_content,
                            metadata={},
                            agent_name=current_agent
                        ))
            continue  # Already incremented i

        # Extract Command outputs (observations/results)
        if commands_output_pattern.search(line):
            # Collect output lines until next log timestamp
            output_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Stop when we hit another log line with timestamp
                if timestamp_pattern.match(next_line):
                    break
                output_lines.append(next_line)
                i += 1

            output_content = '\n'.join(output_lines).strip()
            if output_content:
                steps.append(StandardStep(
                    step_id=len(steps),
                    role="tool",
                    type="observation",
                    content=output_content,
                    metadata={},
                    agent_name=current_agent
                ))
            continue  # Already incremented i

        i += 1

    # Calculate duration with capped gaps
    duration = 0.0
    if len(timestamps) > 1:
        timestamps.sort()
        for j in range(1, len(timestamps)):
            gap = (timestamps[j] - timestamps[j-1]).total_seconds()
            duration += min(gap, MAX_GAP_SECONDS)

    # Determine if multi-agent based on unique agents seen
    # Filter out "User" and count actual agent roles
    actual_agents = {a for a in agents_seen if not a.startswith("User")}
    is_multi_agent = len(actual_agents) > 1

    total_tokens = total_prompt_tokens + total_completion_tokens

    return EvaluationTrace(
        agent_name=agent_name_override,
        task_id=task_id,
        steps=steps,
        total_cost=total_cost,
        total_tokens=total_tokens,
        duration_seconds=duration,
        success_status=None,  # Will be matched later via labels
        is_multi_agent=is_multi_agent
    )


def get_task_description_from_trace(trace: EvaluationTrace) -> str:
    """Extract the task description from the first user instruction step."""
    for step in trace.steps:
        if step.role == "user" and step.type == "instruction":
            return step.content
    return ""


def get_agents_from_trace(trace: EvaluationTrace) -> List[str]:
    """Get list of unique agent names from the trace."""
    agents = set()
    for step in trace.steps:
        if step.agent_name and step.agent_name != "User" and step.agent_name != "unknown":
            agents.add(step.agent_name)
    return sorted(list(agents))
