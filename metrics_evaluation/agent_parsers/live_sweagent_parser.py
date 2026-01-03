"""
Parser for live-swe-agent (mini-swe-agent) trajectory files.

This format uses a chat message structure with:
- "messages": Array of {role: system/user/assistant, content: str}
- "info": Contains model_stats, exit_status, submission (git diff)
- "instance_id": Task identifier

Key differences from standard SWE-Agent:
- Chat format (system/user/assistant) instead of trajectory array
- Task description embedded in first user message with <pr_description> tags
- Tokens tracked per assistant message in extra.response.usage
- Duration derived from Unix timestamps in extra.response.created
"""

import json
import re
from typing import List, Dict, Any, Optional

from evaluation_data_models import EvaluationTrace, StandardStep


def parse_live_sweagent_trajectory(file_path: str, agent_name_override: str = "live-swe-agent") -> EvaluationTrace:
    """
    Parse a live-swe-agent (mini-swe-agent) trajectory file.

    Args:
        file_path: Path to the .traj.json file
        agent_name_override: Name to use for the agent

    Returns:
        EvaluationTrace with standardized steps
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps: List[StandardStep] = []

    # Extract task ID from instance_id or filename
    task_id = data.get("instance_id", "")
    if not task_id:
        import os
        filename = os.path.basename(file_path)
        # Remove .traj.json or .json extension
        task_id = re.sub(r'\.(traj\.)?json$', '', filename)

    # Get messages array
    messages = data.get("messages", [])

    # Track timestamps for duration calculation
    timestamps: List[float] = []

    # Track total tokens
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Extract task description from first user message
    task_description = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        extra = msg.get("extra", {})

        if role == "system":
            # System prompt - typically just configuration, skip as step
            continue

        elif role == "user":
            # First user message contains task description in <pr_description> tags
            if task_description is None:
                pr_match = re.search(r'<pr_description>\s*Consider the following PR description:\s*(.*?)\s*</pr_description>',
                                     content, re.DOTALL)
                if pr_match:
                    task_description = pr_match.group(1).strip()
                    # Add as instruction step
                    steps.append(StandardStep(
                        step_id=len(steps),
                        role="user",
                        type="instruction",
                        content=task_description,
                        metadata={},
                        agent_name="user"
                    ))
                    continue

            # Subsequent user messages are command outputs/observations
            # Parse returncode and output
            returncode_match = re.search(r'<returncode>(\d+)</returncode>', content)
            output_match = re.search(r'<output>\s*(.*?)\s*</output>', content, re.DOTALL)

            # Also handle truncated output format
            if not output_match:
                # Try to get from output_head + output_tail
                head_match = re.search(r'<output_head>\s*(.*?)\s*</output_head>', content, re.DOTALL)
                tail_match = re.search(r'<output_tail>\s*(.*?)\s*</output_tail>', content, re.DOTALL)
                if head_match and tail_match:
                    output_content = head_match.group(1) + "\n...[truncated]...\n" + tail_match.group(1)
                elif head_match:
                    output_content = head_match.group(1)
                else:
                    output_content = content
            else:
                output_content = output_match.group(1) if output_match else content

            metadata = {}
            if returncode_match:
                metadata["returncode"] = int(returncode_match.group(1))

            # Check if this is a format error message (not actual command output)
            if "Please always provide EXACTLY ONE action" in content:
                steps.append(StandardStep(
                    step_id=len(steps),
                    role="user",
                    type="observation",
                    content="[Format error - agent did not provide valid action]",
                    metadata={"error_type": "format_error"},
                    agent_name=agent_name_override
                ))
            else:
                steps.append(StandardStep(
                    step_id=len(steps),
                    role="tool",
                    type="observation",
                    content=output_content,
                    metadata=metadata,
                    agent_name=agent_name_override
                ))

        elif role == "assistant":
            # Assistant messages contain THOUGHT and bash command

            # Extract timestamp for duration calculation
            response = extra.get("response", {})
            created = response.get("created")
            if created:
                timestamps.append(float(created))

            # Extract token usage
            usage = response.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            # Parse THOUGHT section
            thought_match = re.search(r'THOUGHT:\s*(.*?)(?=```|$)', content, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought_content = thought_match.group(1).strip()
                if thought_content:
                    steps.append(StandardStep(
                        step_id=len(steps),
                        role="agent",
                        type="thought",
                        content=thought_content,
                        metadata={},
                        agent_name=agent_name_override
                    ))

            # Parse bash command
            bash_match = re.search(r'```bash\s*(.*?)\s*```', content, re.DOTALL)
            if bash_match:
                action_content = bash_match.group(1).strip()
                steps.append(StandardStep(
                    step_id=len(steps),
                    role="agent",
                    type="action",
                    content=action_content,
                    metadata={},
                    agent_name=agent_name_override
                ))

    # Calculate duration from timestamps
    duration_seconds = 0.0
    if len(timestamps) >= 2:
        # Use first and last timestamp difference
        # Note: This measures wall-clock time for API calls, not actual execution time
        duration_seconds = timestamps[-1] - timestamps[0]

    # Extract cost from model_stats
    info = data.get("info", {})
    model_stats = info.get("model_stats", {})
    total_cost = model_stats.get("instance_cost", 0.0) or 0.0

    # Total tokens
    total_tokens = total_prompt_tokens + total_completion_tokens

    # If no token info from messages, try to estimate from model_stats
    if total_tokens == 0 and model_stats:
        # Some formats have tokens_sent/tokens_received
        tokens_sent = model_stats.get("tokens_sent", 0) or 0
        tokens_received = model_stats.get("tokens_received", 0) or 0
        total_tokens = tokens_sent + tokens_received

    return EvaluationTrace(
        agent_name=agent_name_override,
        task_id=task_id,
        steps=steps,
        total_cost=total_cost,
        total_tokens=total_tokens,
        duration_seconds=duration_seconds,
        success_status=None,  # Will be matched from labels file
        is_multi_agent=False
    )


def get_task_description(trace: EvaluationTrace) -> str:
    """Extract task description from first instruction step."""
    for step in trace.steps:
        if step.type == "instruction":
            return step.content
    return ""
