import argparse
import hashlib
import json
import os
import datetime
import math
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

# Versuch, openai zu importieren
try:
    from openai import OpenAI
    OpenAIClient = OpenAI
except ImportError:
    print("Warnung: 'openai' Paket nicht gefunden. LLM-as-a-Judge Metriken werden übersprungen.")
    OpenAIClient = None

# Try to import sentence-transformers for local embeddings (fallback for metric 5.2)
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDING_MODEL = None  # Lazy loading
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    LOCAL_EMBEDDING_MODEL = None

# Eigene Module importieren
from evaluation_data_models import StandardStep, EvaluationTrace
from evaluation_prompts import PROMPTS

# --- KONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")
# LABELS_FILE is now constructed dynamically based on agent type in main()

BASE_URL_JUDGE = "https://api.helmholtz-blablador.fz-juelich.de/v1"
MODEL_JUDGE = "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
MODEL_EMBEDDING = "text-embedding-3-small"

# Context window size for efficiency calculations (tokens)
# Adjust based on the agent's actual LLM context window
CONTEXT_WINDOW_SIZE = 131000

# --- HELFERFUNKTIONEN ---
def get_agent_parser(agent_type: str):
    if agent_type.lower() == "openhands":
        from agent_parsers.openhands_parser import parse_openhands_trajectory
        return parse_openhands_trajectory
    elif agent_type.lower() == "swe-agent" or agent_type.lower() == "sweagent":
        from agent_parsers.sweagent_parser import parse_sweagent_trajectory
        return parse_sweagent_trajectory
    elif agent_type.lower() == "metagpt":
        from agent_parsers.metagpt_parser import parse_metagpt_trajectory
        return parse_metagpt_trajectory
    elif agent_type.lower() in ("live-swe-agent", "live_swe_agent", "livesweagent", "mini-swe-agent"):
        from agent_parsers.live_sweagent_parser import parse_live_sweagent_trajectory
        return parse_live_sweagent_trajectory
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

# Global list to collect API call logs
API_CALL_LOGS = []

def call_judge(client, prompt, system_msg="You are an expert evaluator.", metric_name="judge_call"):
    if not client: return None

    start_time = time.time()
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "metric": metric_name,
        "model": MODEL_JUDGE,
        "system_message": system_msg,
        "prompt": prompt,
        "prompt_length": len(prompt),
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_JUDGE,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        duration = time.time() - start_time
        content = resp.choices[0].message.content

        # Log usage info if available
        if hasattr(resp, 'usage') and resp.usage:
            log_entry["prompt_tokens"] = resp.usage.prompt_tokens
            log_entry["completion_tokens"] = resp.usage.completion_tokens
            log_entry["total_tokens"] = resp.usage.total_tokens

        log_entry["duration_seconds"] = round(duration, 3)
        log_entry["status"] = "success"

        if content is None:
            log_entry["status"] = "empty_response"
            log_entry["response"] = None
            API_CALL_LOGS.append(log_entry)
            return {"error": "LLM returned empty response"}

        log_entry["response"] = content
        log_entry["response_length"] = len(content)
        API_CALL_LOGS.append(log_entry)
        return json.loads(content)

    except Exception as e:
        duration = time.time() - start_time
        log_entry["duration_seconds"] = round(duration, 3)
        log_entry["status"] = "error"
        log_entry["error"] = str(e)
        log_entry["response"] = None
        API_CALL_LOGS.append(log_entry)
        return {"error": str(e)}

def get_embedding(client, text):
    if not client: return None
    try:
        resp = client.embeddings.create(input=[text], model=MODEL_EMBEDDING)
        return resp.data[0].embedding
    except Exception:
        return None

def get_local_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding using local sentence-transformers model.
    Falls back to this when API embedding is not available.
    """
    global LOCAL_EMBEDDING_MODEL

    if not HAS_SENTENCE_TRANSFORMERS:
        return None

    try:
        # Lazy load the model on first use
        if LOCAL_EMBEDDING_MODEL is None:
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            LOCAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embedding
        embedding = LOCAL_EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Local embedding error: {e}")
        return None

def get_embeddings_batch(texts: List[str], use_local: bool = False) -> List[Optional[List[float]]]:
    """
    Get embeddings for multiple texts, using local model for efficiency.
    """
    global LOCAL_EMBEDDING_MODEL

    if not use_local or not HAS_SENTENCE_TRANSFORMERS:
        return [None] * len(texts)

    try:
        if LOCAL_EMBEDDING_MODEL is None:
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            LOCAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

        # Batch encode for efficiency
        embeddings = LOCAL_EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return [None] * len(texts)

def cosine_similarity(v1, v2):
    dot_product = sum(a*b for a,b in zip(v1, v2))
    norm_a = math.sqrt(sum(a*a for a in v1))
    norm_b = math.sqrt(sum(b*b for b in v2))
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

def get_trace_for_strategy(trace: EvaluationTrace, initial_steps: int = 5, initial_max_chars: int = 2500) -> str:
    """
    Extract trace for strategy evaluation:
    - First N steps after USER instruction: full detail (up to initial_max_chars)
    - Remaining steps: normal pruning logic
    """
    steps = [s for s in trace.steps if s.role != "system"]
    if not steps:
        return "(empty trace)"

    # Find the first USER instruction step
    user_instruction_idx = -1
    for i, s in enumerate(steps):
        if s.role == "user" and s.type == "instruction":
            user_instruction_idx = i
            break

    if user_instruction_idx == -1:
        start_idx = 0
    else:
        start_idx = user_instruction_idx + 1

    # Part 1: First N steps with full content
    first_steps = steps[start_idx:start_idx + initial_steps]
    first_summaries = []
    for s in first_steps:
        first_summaries.append(f"[{s.step_id}] {s.role.upper()} ({s.type}): {s.content.strip()}")

    first_part = "\n".join(first_summaries)
    if len(first_part) > initial_max_chars:
        first_part = first_part[:initial_max_chars] + f"\n... [initial steps truncated at {initial_max_chars} chars]"

    # Part 2: Remaining steps with normal pruning
    remaining_steps = steps[start_idx + initial_steps:]
    if not remaining_steps:
        return first_part

    remaining_summaries = []
    for s in remaining_steps:
        c = s.content.strip()
        # Truncate observations
        if s.type == "observation":
            if len(c) > 200:
                c = c[:100] + f" ... [{len(c)} chars] ... " + c[-50:]
        # Truncate long actions/thoughts
        elif len(c) > 300:
            c = c[:150] + f" ... [{len(c)} chars truncated]"
        remaining_summaries.append(f"[{s.step_id}] {s.role.upper()} ({s.type}): {c}")

    # If too many remaining steps, sample them
    if len(remaining_summaries) > 20:
        n = len(remaining_summaries)
        first_remaining = remaining_summaries[:5]
        last_remaining = remaining_summaries[-5:]
        middle_indices = [int(5 + i * (n - 10) / 5) for i in range(5)]
        middle_remaining = [remaining_summaries[i] for i in middle_indices if 5 <= i < n - 5]

        second_part = "\n".join(first_remaining)
        second_part += f"\n\n... [{n - 10 - len(middle_remaining)} steps omitted] ...\n\n"
        second_part += "\n".join(middle_remaining)
        second_part += f"\n\n... [continuing to final steps] ...\n\n"
        second_part += "\n".join(last_remaining)
    else:
        second_part = "\n".join(remaining_summaries)

    return first_part + "\n\n--- Following Steps ---\n\n" + second_part


def extract_inter_agent_communication(trace: EvaluationTrace, max_chars: int = 12000) -> tuple:
    """
    Extract inter-agent communication messages for MAS metrics.

    Returns:
        tuple: (formatted_log: str, message_count: int, agents_involved: list)

    Inter-agent messages are identified by:
    1. Steps with 'sent_from' and 'send_to' metadata (explicit messages)
    2. Agent thoughts/actions that reference other agents (implicit coordination)
    3. Steps with type 'thought' from different agents communicating

    Excludes:
    - System logs and internal observations
    - Tool execution outputs (unless they're reports to other agents)
    - User instructions (initial task only)
    """
    messages = []
    agents_involved = set()

    for step in trace.steps:
        # Skip system steps and pure tool observations
        if step.role == "system":
            continue

        # Skip user instructions (not inter-agent)
        if step.role == "user" and step.type == "instruction":
            continue

        # Check for explicit inter-agent messages (from metadata)
        sent_from = step.metadata.get("sent_from", "")
        send_to = step.metadata.get("send_to", [])

        # Explicit message between agents
        if sent_from and send_to and sent_from != "User":
            agents_involved.add(sent_from)
            if isinstance(send_to, list):
                for recipient in send_to:
                    if recipient != "<all>" and recipient != "User":
                        agents_involved.add(recipient)
                send_to_str = ", ".join([r for r in send_to if r != "<all>"])
            else:
                send_to_str = send_to
                if send_to != "User":
                    agents_involved.add(send_to)

            content = step.content.strip()
            # Truncate very long messages
            if len(content) > 500:
                content = content[:250] + f" ... [{len(content)} chars] ... " + content[-100:]

            messages.append({
                "step_id": step.step_id,
                "from": sent_from,
                "to": send_to_str,
                "content": content,
                "type": "explicit_message"
            })

        # Agent actions that are communication-related (publish_message, reply_to_human, etc.)
        elif step.role == "agent" and step.type == "action":
            cmd_name = step.metadata.get("command_name", "")
            if any(comm_cmd in cmd_name.lower() for comm_cmd in
                   ["publish", "message", "reply", "send", "delegate", "handoff"]):
                agents_involved.add(step.agent_name)
                content = step.content.strip()
                if len(content) > 500:
                    content = content[:250] + f" ... [{len(content)} chars] ... " + content[-100:]

                messages.append({
                    "step_id": step.step_id,
                    "from": step.agent_name,
                    "to": "(inferred from action)",
                    "content": content,
                    "type": "action_message"
                })

        # Agent thoughts that mention coordination or other agents
        elif step.role == "agent" and step.type == "thought" and step.agent_name != "default":
            agents_involved.add(step.agent_name)
            # Only include if it seems like coordination (mentions other agents or handoff)
            content_lower = step.content.lower()
            coordination_keywords = ["team", "assign", "delegate", "ask", "tell", "inform",
                                    "report", "handoff", "coordinate", "collaborate"]
            if any(kw in content_lower for kw in coordination_keywords):
                content = step.content.strip()
                if len(content) > 300:
                    content = content[:150] + f" ... [{len(content)} chars] ... " + content[-50:]

                messages.append({
                    "step_id": step.step_id,
                    "from": step.agent_name,
                    "to": "(coordination thought)",
                    "content": content,
                    "type": "coordination_thought"
                })

    # Format messages into a readable log
    if not messages:
        return "(No inter-agent communication detected)", 0, list(agents_involved)

    formatted_lines = []
    for i, msg in enumerate(messages, 1):
        formatted_lines.append(
            f"[{i}] Step {msg['step_id']} | {msg['from']} -> {msg['to']}:\n"
            f"    {msg['content']}"
        )

    formatted_log = "\n\n".join(formatted_lines)

    # Truncate if too long
    if len(formatted_log) > max_chars:
        # Keep first and last portions
        half = max_chars // 2 - 100
        formatted_log = (
            formatted_log[:half] +
            f"\n\n... [{len(formatted_log) - max_chars} chars omitted from middle] ...\n\n" +
            formatted_log[-half:]
        )

    return formatted_log, len(messages), sorted(list(agents_involved))


def calculate_metric_5_1_communication_efficiency(trace: EvaluationTrace, client, task_desc: str):
    """
    Metric 5.1: Multi-Agent Communication Efficiency

    Evaluates the signal-to-noise ratio in inter-agent communication.
    Uses LLM-as-a-Judge to analyze communication patterns.
    """
    if not trace.is_multi_agent:
        return "N/A - Single Agent"

    # Extract inter-agent communication
    communication_log, message_count, agents_involved = extract_inter_agent_communication(trace)

    if message_count == 0:
        return {
            "score": None,
            "note": "No inter-agent communication detected in trace",
            "agents_in_trace": agents_involved,
            "method": "llm_as_a_judge"
        }

    if not client:
        return {
            "note": "Skipped (No LLM Client)",
            "message_count": message_count,
            "agents_involved": agents_involved
        }

    # Build prompt with extracted communication
    prompt = PROMPTS["5.1"].format(
        task=task_desc,
        communication_log=communication_log
    )

    result = call_judge(client, prompt, metric_name="metric_5.1_communication_efficiency")

    # Add metadata to result
    if isinstance(result, dict) and "error" not in result:
        result["_metadata"] = {
            "messages_analyzed": message_count,
            "agents_involved": agents_involved,
            "communication_log_chars": len(communication_log)
        }

    return result


def prune_trace(trace: EvaluationTrace, max_chars: int = 15000) -> str:
    """Summarize trace for LLM evaluation, keeping it under max_chars."""
    steps = [s for s in trace.steps if s.role != "system"]
    if not steps:
        return "(empty trace)"

    # First pass: create condensed step summaries
    summaries = []
    for s in steps:
        c = s.content.strip()
        # Heavily truncate observations (tool outputs)
        if s.type == "observation":
            if len(c) > 200:
                c = c[:100] + f" ... [{len(c)} chars] ... " + c[-50:]
        # Truncate long actions/thoughts
        elif len(c) > 300:
            c = c[:150] + f" ... [{len(c)} chars truncated]"
        summaries.append(f"[{s.step_id}] {s.role.upper()} ({s.type}): {c}")

    # If still too large, sample steps: keep first 10, last 10, and sample middle
    total_len = sum(len(s) for s in summaries)
    if total_len > max_chars and len(summaries) > 30:
        n_steps = len(summaries)
        first_steps = summaries[:10]
        last_steps = summaries[-10:]
        middle_count = max(5, (max_chars - sum(len(s) for s in first_steps + last_steps)) // 200)
        middle_indices = [int(10 + i * (n_steps - 20) / middle_count) for i in range(middle_count)]
        middle_steps = [summaries[i] for i in middle_indices if 10 <= i < n_steps - 10]

        out = "\n".join(first_steps)
        out += f"\n\n... [{n_steps - 20 - len(middle_steps)} steps omitted] ...\n\n"
        out += "\n".join(middle_steps)
        out += f"\n\n... [continuing to final steps] ...\n\n"
        out += "\n".join(last_steps)
        return out

    return "\n".join(summaries)

# --- METRIK IMPLEMENTIERUNGEN ---

def calculate_metric_2_1_loop_detection(trace: EvaluationTrace):
    sequence = []
    for s in trace.steps:
        if s.type in ["action", "observation"]:
            normalized = f"{s.type}:{s.content.strip()[:100]}" 
            sequence.append(normalized)

    min_window = 2
    max_window = 5
    threshold = 3 

    detected_loops = []
    
    for window_size in range(min_window, max_window + 1):
        seen_hashes = defaultdict(list)
        for i in range(len(sequence) - window_size + 1):
            window = tuple(sequence[i : i + window_size])
            window_hash = hashlib.md5(str(window).encode('utf-8')).hexdigest()
            seen_hashes[window_hash].append(i)

        for h, indices in seen_hashes.items():
            if len(indices) >= threshold:
                detected_loops.append({
                    "window_size": window_size,
                    "count": len(indices),
                    "example": sequence[indices[0] : indices[0] + window_size]
                })

    return {"loop_detected": len(detected_loops) > 0, "details": detected_loops}

def calculate_metric_3_3_tool_usage_efficiency(trace: EvaluationTrace, metric_3_1_results: Optional[Dict] = None, context_window_size: int = CONTEXT_WINDOW_SIZE):
    """
    Calculate Tool Usage Efficiency based on context pollution analysis.

    This metric measures how efficiently the agent uses tools by considering:
    1. The "future cost" of each observation (tokens × remaining steps)
    2. Quality weights from metric 3.1 (suboptimal selections are penalized more)
    3. Context window truncation (very large outputs are capped)

    A higher efficiency_score (0-1) is better.
    """
    # Get all observations
    observations = [(i, s) for i, s in enumerate(trace.steps) if s.type == "observation"]
    N = len(observations)

    if N == 0:
        return {
            "efficiency_score": 1.0,
            "raw_context_cost": 0,
            "weighted_context_cost": 0,
            "context_window_size": context_window_size,
            "total_observations": 0,
            "worst_offender": None,
            "note": "No observations found",
            "method": "deterministic_calculation"
        }

    # Quality weights based on 3.1 tool selection quality
    quality_weights = {
        "Optimal": 1.0,
        "Good": 1.0,
        "Suboptimal": 1.5,  # 50% penalty for avoidable waste
        "Poor": 2.0,       # Double penalty for poor choices
        "Hallucinated": 2.0
    }

    # Build lookup from 3.1 results (map trace_step_id -> quality)
    quality_lookup = {}
    if metric_3_1_results and isinstance(metric_3_1_results, dict):
        for step in metric_3_1_results.get("steps", []):
            trace_step_id = step.get("trace_step_id")
            quality = step.get("selection_quality", "Unknown")
            if trace_step_id is not None:
                quality_lookup[trace_step_id] = quality

    # Cap for truncation: observations larger than this are likely truncated in practice
    MAX_USEFUL_TOKENS = min(context_window_size // 4, 20000)

    # Approximate token count (rough estimate: 1 token ≈ 4 chars for English text)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    step_details = []
    raw_context_cost = 0
    weighted_context_cost = 0
    total_tokens = 0
    worst_offender = None

    for obs_idx, (trace_idx, obs_step) in enumerate(observations):
        tokens = estimate_tokens(obs_step.content)
        total_tokens += tokens
        remaining_steps = N - obs_idx - 1

        # Find the action that caused this observation (typically previous step)
        action_idx = trace_idx - 1
        action_content = ""
        if action_idx >= 0 and action_idx < len(trace.steps):
            action_step = trace.steps[action_idx]
            if action_step.type == "action":
                action_content = action_step.content[:100]

        # Get quality weight from 3.1 (use action_idx to look up)
        quality = quality_lookup.get(action_idx, "Unknown")
        weight = quality_weights.get(quality, 1.0)

        # Apply truncation cap for effective tokens
        effective_tokens = min(tokens, MAX_USEFUL_TOKENS)

        # Calculate costs
        raw_step_cost = tokens * remaining_steps
        weighted_step_cost = effective_tokens * remaining_steps * weight

        raw_context_cost += raw_step_cost
        weighted_context_cost += weighted_step_cost

        step_detail = {
            "observation_index": obs_idx,
            "trace_step_id": trace_idx,
            "tokens": tokens,
            "effective_tokens": effective_tokens,
            "remaining_steps": remaining_steps,
            "quality_from_3_1": quality,
            "quality_weight": weight,
            "raw_cost": raw_step_cost,
            "weighted_cost": weighted_step_cost
        }

        if action_content:
            step_detail["action"] = action_content + ("..." if len(action_content) >= 100 else "")

        step_details.append(step_detail)

        # Track worst offender
        current_worst_cost = worst_offender["weighted_cost"] if worst_offender else 0
        if weighted_step_cost > current_worst_cost:
            worst_offender = {
                "observation_index": obs_idx,
                "trace_step_id": trace_idx,
                "tokens": tokens,
                "effective_tokens": effective_tokens,
                "quality": quality,
                "weighted_cost": weighted_step_cost,
                "action": action_content + ("..." if len(action_content) >= 100 else "")
            }

    # Normalize to 0-1 score
    # Theoretical worst case: all tokens at step 0 with Poor quality (weight=2.0)
    max_possible_cost = total_tokens * N * 2.0 if total_tokens > 0 else 1
    efficiency_score = 1.0 - (weighted_context_cost / max_possible_cost)
    efficiency_score = max(0.0, min(1.0, efficiency_score))

    return {
        "efficiency_score": round(efficiency_score, 4),
        "raw_context_cost": raw_context_cost,
        "weighted_context_cost": weighted_context_cost,
        "context_window_size": context_window_size,
        "max_useful_tokens_cap": MAX_USEFUL_TOKENS,
        "total_observations": N,
        "total_observation_tokens": total_tokens,
        "worst_offender": worst_offender,
        "assumptions": [
            "Token count estimated as len(text)/4",
            f"Observations capped at {MAX_USEFUL_TOKENS} tokens (truncation assumption)",
            "Quality weights: Optimal/Good=1.0, Suboptimal=1.5, Poor/Hallucinated=2.0",
            "Non-sampled steps from 3.1 assumed quality_weight=1.0"
        ],
        "step_details": step_details,
        "method": "deterministic_calculation_with_3.1_weighting"
    }

def extract_message_contents_for_diversity(trace: EvaluationTrace) -> list:
    """
    Extract message contents specifically for diversity analysis.

    Returns a list of dicts with message content and metadata for embedding.
    Focuses on substantive inter-agent communication content.
    """
    messages = []

    for step in trace.steps:
        # Skip system and user steps
        if step.role in ["system", "user"]:
            continue

        # Skip pure tool observations (these are command outputs, not agent communication)
        if step.role == "tool":
            continue

        # Focus on agent communication steps
        if step.role == "agent" and step.agent_name != "default":
            content = step.content.strip()

            # Skip empty or very short messages
            if len(content) < 20:
                continue

            # Check if this is inter-agent communication
            sent_from = step.metadata.get("sent_from", "")
            send_to = step.metadata.get("send_to", [])
            cmd_name = step.metadata.get("command_name", "")

            is_communication = False

            # Explicit messages between agents
            if sent_from and send_to:
                is_communication = True

            # Communication-related actions
            elif step.type == "action" and any(kw in cmd_name.lower() for kw in
                ["publish", "message", "reply", "send", "delegate"]):
                is_communication = True

            # Agent thoughts/coordination
            elif step.type == "thought":
                is_communication = True

            if is_communication:
                # Truncate very long messages for embedding efficiency
                if len(content) > 1500:
                    content = content[:750] + " ... " + content[-750:]

                messages.append({
                    "step_id": step.step_id,
                    "agent": step.agent_name,
                    "content": content,
                    "type": step.type
                })

    return messages


def calculate_metric_5_2_information_diversity(trace: EvaluationTrace, client):
    """
    Metric 5.2: Information Diversity Score (IDS)

    Measures the semantic diversity of information exchanged between agents.
    Based on the GEMMAS paper approach but simplified (no spatial/temporal adjacency matrices).

    Formula: IDS = 1 - avg(pairwise_cosine_similarity)

    Interpretation:
    - 1.0: Maximum diversity (agents discuss completely different topics)
    - 0.0: No diversity (agents repeat the same information)

    Uses local embeddings (sentence-transformers) as primary method for efficiency.
    Falls back to API embeddings if local model is not available.

    Args:
        trace: The evaluation trace containing agent steps
        client: OpenAI client for embeddings (optional fallback)

    Returns:
        dict with score, metadata, and method
    """
    if not trace.is_multi_agent:
        return "N/A - Single Agent"

    # Extract messages for diversity analysis
    messages = extract_message_contents_for_diversity(trace)

    if len(messages) < 2:
        return {
            "score": None,
            "note": f"Insufficient messages for diversity analysis (found {len(messages)}, need >= 2)",
            "messages_found": len(messages),
            "method": "embedding_similarity"
        }

    # Limit messages for efficiency (sample if too many)
    max_messages = 30
    if len(messages) > max_messages:
        # Sample evenly across the conversation
        step = len(messages) // max_messages
        sampled_messages = [messages[i] for i in range(0, len(messages), step)][:max_messages]
        sampling_note = f"Sampled {len(sampled_messages)} of {len(messages)} messages"
    else:
        sampled_messages = messages
        sampling_note = None

    # Extract texts for embedding
    texts = [msg["content"] for msg in sampled_messages]
    embedding_source = None

    # Try local embeddings first (more reliable and efficient)
    embeddings = []
    embedding_metadata = []

    if HAS_SENTENCE_TRANSFORMERS:
        print("Running 5.2 Information Diversity (Local Embeddings)...")
        batch_embeddings = get_embeddings_batch(texts, use_local=True)
        for i, (msg, emb) in enumerate(zip(sampled_messages, batch_embeddings)):
            if emb is not None:
                embeddings.append(emb)
                embedding_metadata.append({
                    "step_id": msg["step_id"],
                    "agent": msg["agent"],
                    "content_length": len(msg["content"])
                })
        if embeddings:
            embedding_source = "local_sentence_transformers"

    # Fallback to API embeddings if local failed
    if len(embeddings) < 2 and client:
        print("Falling back to API embeddings for 5.2...")
        embeddings = []
        embedding_metadata = []
        for msg in sampled_messages:
            embedding = get_embedding(client, msg["content"])
            if embedding:
                embeddings.append(embedding)
                embedding_metadata.append({
                    "step_id": msg["step_id"],
                    "agent": msg["agent"],
                    "content_length": len(msg["content"])
                })
        if embeddings:
            embedding_source = "api_openai"

    # Check if we have enough embeddings
    if len(embeddings) < 2:
        note = "Could not generate enough embeddings"
        if not HAS_SENTENCE_TRANSFORMERS:
            note += " (sentence-transformers not installed, run: pip install sentence-transformers)"
        elif not client:
            note += " (no API client and local embeddings failed)"

        return {
            "score": None,
            "note": note,
            "messages_attempted": len(sampled_messages),
            "method": "embedding_similarity"
        }

    # Calculate pairwise cosine similarities
    n = len(embeddings)
    similarities = []
    similarity_details = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

            # Track highest similarities (most redundant pairs)
            if sim > 0.85:  # High similarity threshold
                similarity_details.append({
                    "pair": [embedding_metadata[i]["step_id"], embedding_metadata[j]["step_id"]],
                    "agents": [embedding_metadata[i]["agent"], embedding_metadata[j]["agent"]],
                    "similarity": round(sim, 3)
                })

    # Calculate Information Diversity Score
    # IDS = 1 - average_similarity
    avg_similarity = sum(similarities) / len(similarities)
    ids_score = 1.0 - avg_similarity

    # Calculate per-agent contribution to diversity
    agent_messages = {}
    for meta in embedding_metadata:
        agent = meta["agent"]
        if agent not in agent_messages:
            agent_messages[agent] = 0
        agent_messages[agent] += 1

    # Sort redundant pairs by similarity (most redundant first)
    similarity_details.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "score": round(ids_score, 4),
        "average_similarity": round(avg_similarity, 4),
        "num_messages_analyzed": len(embeddings),
        "num_similarity_pairs": len(similarities),
        "similarity_stats": {
            "min": round(min(similarities), 4),
            "max": round(max(similarities), 4),
            "std": round((sum((s - avg_similarity)**2 for s in similarities) / len(similarities))**0.5, 4)
        },
        "messages_per_agent": agent_messages,
        "high_similarity_pairs": similarity_details[:5],  # Top 5 most redundant pairs
        "sampling_note": sampling_note,
        "embedding_source": embedding_source,
        "interpretation": (
            "High diversity" if ids_score >= 0.7 else
            "Moderate diversity" if ids_score >= 0.4 else
            "Low diversity (high redundancy)"
        ),
        "method": "embedding_similarity"
    }

def extract_speaker_sequence(trace: EvaluationTrace) -> list:
    """
    Extract the chronological sequence of speakers (agent IDs) from the trace.

    Only includes steps where an agent actively communicates or acts.
    Consecutive steps by the same agent are collapsed into a single entry
    to focus on turn-taking patterns.

    Returns:
        List of dicts with agent name and step range
    """
    speakers = []

    for step in trace.steps:
        # Skip non-agent steps
        if step.role not in ["agent"]:
            continue

        # Skip default/unknown agents
        if step.agent_name in ["default", "unknown", ""]:
            continue

        # Add to sequence (will be collapsed later)
        speakers.append({
            "agent": step.agent_name,
            "step_id": step.step_id,
            "type": step.type
        })

    if not speakers:
        return []

    # Collapse consecutive same-agent entries into turns
    turns = []
    current_turn = {
        "agent": speakers[0]["agent"],
        "start_step": speakers[0]["step_id"],
        "end_step": speakers[0]["step_id"],
        "num_actions": 1
    }

    for i in range(1, len(speakers)):
        if speakers[i]["agent"] == current_turn["agent"]:
            # Same agent continues
            current_turn["end_step"] = speakers[i]["step_id"]
            current_turn["num_actions"] += 1
        else:
            # New agent takes over - save current turn and start new one
            turns.append(current_turn)
            current_turn = {
                "agent": speakers[i]["agent"],
                "start_step": speakers[i]["step_id"],
                "end_step": speakers[i]["step_id"],
                "num_actions": 1
            }

    # Don't forget the last turn
    turns.append(current_turn)

    return turns


def calculate_metric_5_3_path_redundancy(trace: EvaluationTrace):
    """
    Metric 5.3: Unique Path Redundancy (Simplified UPR)

    Analyzes communication structure for inefficient "ping-pong" patterns.
    A ping-pong occurs when agent A hands off to B, then B immediately
    hands back to A (A -> B -> A pattern).

    Such patterns often indicate:
    - Stagnation in problem-solving
    - Unclear task delegation
    - Missing information requiring back-and-forth
    - Lack of agent autonomy

    Formula: Score = 1 - (ping_pong_count / total_transitions)

    Interpretation:
    - 1.0: Forward-flowing communication (no unnecessary back-and-forth)
    - 0.5-0.9: Some ping-pong patterns but generally productive
    - < 0.5: High ping-pong ratio (potential stagnation)

    Returns:
        dict with score, detailed metrics, and detected patterns
    """
    if not trace.is_multi_agent:
        return "N/A - Single Agent"

    # Extract speaker sequence (collapsed turns)
    turns = extract_speaker_sequence(trace)

    if len(turns) < 2:
        return {
            "score": 1.0,
            "note": f"Insufficient turns for analysis (found {len(turns)})",
            "total_turns": len(turns),
            "total_transitions": 0,
            "ping_pong_count": 0,
            "method": "simplified_interaction_loop_counting"
        }

    # Extract just the agent names in order
    agent_sequence = [t["agent"] for t in turns]

    # Count transitions (speaker changes)
    # A transition occurs between every consecutive pair of different speakers
    total_transitions = len(turns) - 1  # Since turns are already collapsed

    # Detect ping-pong patterns (A -> B -> A)
    ping_pong_loops = []
    ping_pong_count = 0

    for i in range(len(agent_sequence) - 2):
        a1 = agent_sequence[i]
        a2 = agent_sequence[i + 1]
        a3 = agent_sequence[i + 2]

        # Ping-pong: A -> B -> A (different agent in middle, return to original)
        if a1 == a3 and a1 != a2:
            ping_pong_count += 1
            ping_pong_loops.append({
                "pattern": f"{a1} -> {a2} -> {a1}",
                "turn_indices": [i, i + 1, i + 2],
                "step_ranges": [
                    f"{turns[i]['start_step']}-{turns[i]['end_step']}",
                    f"{turns[i+1]['start_step']}-{turns[i+1]['end_step']}",
                    f"{turns[i+2]['start_step']}-{turns[i+2]['end_step']}"
                ]
            })

    # Calculate score
    if total_transitions > 0:
        score = 1.0 - (ping_pong_count / total_transitions)
    else:
        score = 1.0

    # Analyze flow patterns
    # Count unique agent pairs in transitions
    transition_pairs = Counter()
    for i in range(len(agent_sequence) - 1):
        pair = (agent_sequence[i], agent_sequence[i + 1])
        transition_pairs[pair] += 1

    # Find most common transitions
    common_transitions = [
        {"from": pair[0], "to": pair[1], "count": count}
        for pair, count in transition_pairs.most_common(5)
    ]

    # Identify dominant communication pattern
    if len(set(agent_sequence)) == 2:
        flow_pattern = "Bilateral (2 agents)"
    elif ping_pong_count > total_transitions * 0.3:
        flow_pattern = "High back-and-forth"
    elif len(set(agent_sequence)) >= 3 and ping_pong_count < total_transitions * 0.1:
        flow_pattern = "Multi-agent forward flow"
    else:
        flow_pattern = "Mixed"

    return {
        "score": round(score, 4),
        "total_turns": len(turns),
        "total_transitions": total_transitions,
        "ping_pong_count": ping_pong_count,
        "ping_pong_ratio": round(ping_pong_count / total_transitions, 4) if total_transitions > 0 else 0,
        "unique_agents": len(set(agent_sequence)),
        "agent_sequence_summary": agent_sequence if len(agent_sequence) <= 20 else
            agent_sequence[:10] + ["..."] + agent_sequence[-5:],
        "common_transitions": common_transitions,
        "ping_pong_details": ping_pong_loops[:10],  # Limit to first 10
        "flow_pattern": flow_pattern,
        "interpretation": (
            "Excellent flow (minimal back-and-forth)" if score >= 0.9 else
            "Good flow (occasional ping-pong)" if score >= 0.7 else
            "Moderate redundancy" if score >= 0.5 else
            "High redundancy (frequent ping-pong patterns)"
        ),
        "method": "simplified_interaction_loop_counting"
    }

def calculate_gini_coefficient(values: list) -> float:
    """
    Calculate the Gini coefficient for a list of values.

    The Gini coefficient measures inequality in a distribution:
    - 0.0: Perfect equality (everyone has the same)
    - 1.0: Maximum inequality (one has everything)

    Uses the formula: G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    where x_i are sorted values and i is the 1-based index.
    """
    if not values or len(values) < 2:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)

    if total == 0:
        return 0.0

    # Calculate using the formula for Gini coefficient
    cumulative_sum = sum((i + 1) * x for i, x in enumerate(sorted_values))
    gini = (2 * cumulative_sum) / (n * total) - (n + 1) / n

    return max(0.0, min(1.0, gini))  # Clamp to [0, 1]


def calculate_metric_5_4_agent_invocation(trace: EvaluationTrace):
    """
    Metric 5.4: Agent Invocation Distribution

    Measures the distribution of workload (activity) between different agent roles.
    Uses Shannon entropy to quantify how evenly work is distributed.

    This metric identifies:
    - Balanced teams (high score): All agents contribute roughly equally
    - Dominated teams (low score): One or few agents do most of the work
    - "Pareto problem": When 20% of agents do 80% of the work

    Formula:
    - Count actions per agent: c_i
    - Calculate probability: p_i = c_i / sum(c)
    - Shannon entropy: H = -sum(p_i * log2(p_i))
    - Normalized score: H / H_max where H_max = log2(num_agents)

    Also calculates Gini coefficient as supplementary inequality measure.

    Interpretation:
    - Score 1.0: Perfect equal distribution
    - Score 0.7-0.9: Reasonably balanced
    - Score 0.4-0.6: Moderate imbalance
    - Score < 0.4: Highly unequal (potential bottleneck)

    Returns:
        dict with entropy score, distribution stats, and Gini coefficient
    """
    if not trace.is_multi_agent:
        return "N/A - Single Agent"

    # Count actions per agent (only agent role, exclude default/unknown)
    agent_steps = [s for s in trace.steps
                   if s.role == "agent" and s.agent_name not in ["default", "unknown", ""]]

    if not agent_steps:
        return {
            "score": None,
            "note": "No agent actions found in trace",
            "method": "shannon_entropy"
        }

    # Count by agent
    agent_counts = Counter(s.agent_name for s in agent_steps)
    total_actions = sum(agent_counts.values())
    num_agents = len(agent_counts)

    if num_agents < 2:
        return {
            "score": None,
            "note": f"Only {num_agents} agent(s) found, need >= 2 for distribution analysis",
            "distribution": dict(agent_counts),
            "method": "shannon_entropy"
        }

    # Calculate probability distribution
    probabilities = {agent: count / total_actions for agent, count in agent_counts.items()}

    # Calculate Shannon entropy
    entropy = 0.0
    for p in probabilities.values():
        if p > 0:
            entropy -= p * math.log2(p)

    # Maximum entropy (uniform distribution)
    max_entropy = math.log2(num_agents)

    # Normalized entropy score
    entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

    # Calculate Gini coefficient
    gini = calculate_gini_coefficient(list(agent_counts.values()))

    # Calculate percentages
    percentages = {agent: round(p * 100, 1) for agent, p in probabilities.items()}

    # Find dominant agent (if any)
    sorted_agents = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_agent = sorted_agents[0]
    least_active_agent = sorted_agents[-1]

    # Check for Pareto-like distribution (top agent does > 50%)
    is_pareto = probabilities[dominant_agent[0]] > 0.5

    # Detailed distribution with counts and percentages
    distribution_details = [
        {
            "agent": agent,
            "actions": count,
            "percentage": round(count / total_actions * 100, 1)
        }
        for agent, count in sorted_agents
    ]

    # Determine balance interpretation
    if entropy_score >= 0.9:
        balance_assessment = "Excellent balance (near-equal distribution)"
    elif entropy_score >= 0.7:
        balance_assessment = "Good balance (reasonably distributed)"
    elif entropy_score >= 0.5:
        balance_assessment = "Moderate imbalance"
    elif entropy_score >= 0.3:
        balance_assessment = "Significant imbalance (consider redistribution)"
    else:
        balance_assessment = "Severe imbalance (single agent dominance)"

    return {
        "score": round(entropy_score, 4),
        "entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "gini_coefficient": round(gini, 4),
        "num_agents": num_agents,
        "total_actions": total_actions,
        "distribution": dict(agent_counts),
        "distribution_percentages": percentages,
        "distribution_details": distribution_details,
        "dominant_agent": {
            "name": dominant_agent[0],
            "actions": dominant_agent[1],
            "percentage": round(dominant_agent[1] / total_actions * 100, 1)
        },
        "least_active_agent": {
            "name": least_active_agent[0],
            "actions": least_active_agent[1],
            "percentage": round(least_active_agent[1] / total_actions * 100, 1)
        },
        "is_pareto_distribution": is_pareto,
        "balance_assessment": balance_assessment,
        "interpretation": (
            f"Work is {'evenly' if entropy_score >= 0.7 else 'unevenly'} distributed. "
            f"{dominant_agent[0]} is most active ({percentages[dominant_agent[0]]}%), "
            f"{least_active_agent[0]} is least active ({percentages[least_active_agent[0]]}%)."
        ),
        "method": "shannon_entropy"
    }


# --- MAIN ---
def main(file_path, agent_type_arg, force_mas_arg, sample_rate: int = 5, context_window_steps: int = 8, context_sample_rate: int = 4):
    global API_CALL_LOGS
    API_CALL_LOGS = []  # Clear logs for each run
    print(f"\n{'='*60}")
    print(f"METRICS EVALUATION")
    print(f"{'='*60}")
    print(f"LLM Judge Base URL: {BASE_URL_JUDGE}")
    print(f"LLM Judge Model:    {MODEL_JUDGE}")
    print(f"{'='*60}")
    print(f"\n--- Processing {file_path} (Agent: {agent_type_arg}) ---")
    labels = {}
    labels_file = os.path.join(SCRIPT_DIR, f"evaluation_labels_{agent_type_arg}.json")
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f: labels = json.load(f)

    # Parser aufrufen mit agent_type als Name
    parser_func = get_agent_parser(agent_type_arg)
    trace = parser_func(file_path, agent_type_arg)
    
    if force_mas_arg: trace.is_multi_agent = True
    
    print(f"Parsed {len(trace.steps)} steps. Task: {trace.task_id}. MAS: {trace.is_multi_agent}")

    llm_key = os.getenv("HELMHOLTZ_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = None
    if OpenAIClient and llm_key:
        client = OpenAIClient(base_url=BASE_URL_JUDGE, api_key=llm_key)
        print(f"LLM Client: Connected")
    else:
        print(f"LLM Client: Not available (missing API key or openai package)")

    # --- RESULTS CONSTRUCTION (STRICT ORDER) ---
    results = {
        "meta": {
            "agent": trace.agent_name,
            "task": trace.task_id,
            "timestamp": str(datetime.datetime.now()),
            "is_multi_agent_system": trace.is_multi_agent,
            "llm_judge_model": MODEL_JUDGE
        }
    }

    # Kategorie 1: Ergebnis & Kosten (Hard Facts)
    # Metrik 1.1: Task Success Rate
    is_suc = labels.get(trace.task_id)
    results["metric_1_1_task_success_rate"] = {"success": is_suc, "source": "manual_labels"}

    # Metrik 1.2: Resource Efficiency
    results["metric_1_2_resource_efficiency"] = {
        "total_cost_usd": trace.total_cost, 
        "total_tokens": trace.total_tokens, 
        "duration_seconds": trace.duration_seconds,
        "step_count": len(trace.steps)
    }

    # Kategorie 2: Strategie & Navigation (Der Weg)
    # Metrik 2.1: Loop Detection (Deterministisch)
    results["metric_2_1_loop_detection"] = calculate_metric_2_1_loop_detection(trace)

    # Metrik 2.2: Trajectory Efficiency (Judge)
    # Extract task description from first user message, truncate if too long
    raw_task = next((s.content for s in trace.steps if s.role == "user"), "Unknown Task")
    task_desc = raw_task[:1000] + "..." if len(raw_task) > 1000 else raw_task
    pruned = prune_trace(trace)
    if client:
        print("Running 2.2 Trajectory Efficiency (LLM)...")
        results["metric_2_2_trajectory_efficiency"] = call_judge(client, PROMPTS["2.2"].format(task=task_desc, log=pruned), metric_name="metric_2.2_trajectory_efficiency")
    else:
        results["metric_2_2_trajectory_efficiency"] = "Skipped (No LLM Client)"

    # Metrik 2.3: Global Strategy Consistency (Judge)
    # Use specialized function: first 5 steps full detail (2500 chars), then remaining steps with normal pruning
    if client:
        print("Running 2.3 Global Strategy Consistency (LLM)...")
        strategy_trace = get_trace_for_strategy(trace)
        results["metric_2_3_global_strategy_consistency"] = call_judge(client, PROMPTS["2.3"].format(task=task_desc, log=strategy_trace), metric_name="metric_2.3_global_strategy_consistency")
    else:
        results["metric_2_3_global_strategy_consistency"] = "Skipped (No LLM Client)"

    # Metrik 2.4: Stepwise Reasoning Quality (Judge - Batched, ~15 steps per call)
    if client:
        # Find all action steps
        action_indices = [i for i, s in enumerate(trace.steps) if s.type == "action"]
        batch_size = 15
        num_batches = (len(action_indices) + batch_size - 1) // batch_size if action_indices else 0
        print(f"Running 2.4 Stepwise Reasoning Quality (LLM) - {len(action_indices)} action steps in {num_batches} batch(es)...")

        if action_indices:
            all_step_results = []
            all_scores = []
            total_flaws = 0

            # Process in batches
            for batch_num in range(num_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(action_indices))
                batch_indices = action_indices[batch_start:batch_end]

                steps_text = ""
                for i in batch_indices:
                    s = trace.steps[i]
                    # Get previous context (up to 300 chars)
                    if i > 0:
                        prev_step = trace.steps[i-1]
                        prev_context = prev_step.content[:300]
                        if len(prev_step.content) > 300:
                            prev_context += "..."
                    else:
                        prev_context = "(first step)"

                    # Get action content (up to 300 chars)
                    action_content = s.content[:300]
                    if len(s.content) > 300:
                        action_content += "..."

                    # Get result (next step, up to 300 chars)
                    if i + 1 < len(trace.steps):
                        next_step = trace.steps[i+1]
                        result_content = next_step.content[:300]
                        if len(next_step.content) > 300:
                            result_content += "..."
                    else:
                        result_content = "(last step)"

                    steps_text += f"\n--- Step {i} ---\n"
                    steps_text += f"Previous Context: {prev_context}\n"
                    steps_text += f"Action: {action_content}\n"
                    steps_text += f"Result: {result_content}\n"

                batch_result = call_judge(
                    client,
                    PROMPTS["2.4"].format(steps=steps_text),
                    metric_name=f"metric_2.4_batch_{batch_num + 1}_of_{num_batches}"
                )

                # Aggregate results from this batch
                if isinstance(batch_result, dict) and "steps" in batch_result:
                    all_step_results.extend(batch_result.get("steps", []))
                    total_flaws += batch_result.get("total_flaws", 0)
                    if batch_result.get("overall_score") is not None:
                        all_scores.append(batch_result["overall_score"])

            # Calculate overall score across all batches
            overall_score = sum(all_scores) / len(all_scores) if all_scores else None

            results["metric_2_4_stepwise_reasoning_quality"] = {
                "overall_score": overall_score,
                "total_flaws": total_flaws,
                "total_steps_evaluated": len(action_indices),
                "num_batches": num_batches,
                "steps": all_step_results,
                "method": "llm_as_a_judge"
            }
        else:
            results["metric_2_4_stepwise_reasoning_quality"] = {"overall_score": None, "total_flaws": 0, "steps": [], "note": "No action steps found"}
    else:
        results["metric_2_4_stepwise_reasoning_quality"] = "Skipped (No LLM Client)"

    # Metrik 2.5: Role Adherence (Judge)
    if client:
        print("Running 2.5 Role Adherence (LLM)...")
        results["metric_2_5_role_adherence"] = call_judge(client, PROMPTS["2.5"].format(task=task_desc, log=pruned), metric_name="metric_2.5_role_adherence")
    else:
        results["metric_2_5_role_adherence"] = "Skipped (No LLM Client)"

    # Kategorie 3: Werkzeuge (Die Handlungen)
    # Metrik 3.1: Tool Selection Quality (Judge - Sampled)
    if client:
        # Get ALL action steps first, then sample every Nth
        all_action_indices = [i for i, s in enumerate(trace.steps) if s.type == "action"]
        sampled_indices = all_action_indices[::sample_rate]  # [0th, Nth, 2Nth action, ...]

        total_actions = len(all_action_indices)
        print(f"Running 3.1 Tool Selection Quality (LLM) - {len(sampled_indices)} of {total_actions} actions (every {sample_rate}th)...")

        if sampled_indices:
            all_step_results = []
            all_scores = []
            quality_counts = {"Optimal": 0, "Good": 0, "Suboptimal": 0, "Poor": 0, "Hallucinated": 0}

            for action_num, trace_idx in enumerate(sampled_indices):
                s = trace.steps[trace_idx]
                action_sequence_num = all_action_indices.index(trace_idx) + 1  # 1-based action number

                # Get preceding context (thought or previous observation, up to 500 chars)
                ctx = ""
                if trace_idx > 0:
                    prev = trace.steps[trace_idx - 1]
                    ctx = f"[{prev.type}] {prev.content[:500]}"
                    if len(prev.content) > 500:
                        ctx += "..."
                else:
                    ctx = "(first step)"

                # Get action content
                action_content = s.content[:500]
                if len(s.content) > 500:
                    action_content += "..."

                # Get observation/result if next step is observation (action-observation pair)
                result_preview = ""
                if trace_idx + 1 < len(trace.steps):
                    next_step = trace.steps[trace_idx + 1]
                    if next_step.type == "observation":
                        result_preview = next_step.content[:300]
                        if len(next_step.content) > 300:
                            result_preview += f"... [{len(next_step.content)} chars total]"

                # Build enhanced context with result
                full_context = ctx
                if result_preview:
                    full_context += f"\n\n[Result of this action]: {result_preview}"

                step_result = call_judge(
                    client,
                    PROMPTS["3.1"].format(
                        task=task_desc,
                        step_id=action_sequence_num,  # Use action sequence number
                        context=full_context,
                        action=action_content
                    ),
                    metric_name=f"metric_3.1_action_{action_sequence_num}_trace_step_{trace_idx}"
                )

                if isinstance(step_result, dict):
                    # Add clear identification
                    step_result["action_number"] = action_sequence_num  # 1st action, 5th action, etc.
                    step_result["trace_step_id"] = trace_idx  # Actual step ID in trace
                    if "action_evaluated" not in step_result:
                        step_result["action_evaluated"] = s.content[:100] + ("..." if len(s.content) > 100 else "")

                    all_step_results.append(step_result)

                    if step_result.get("score") is not None:
                        all_scores.append(step_result["score"])

                    quality = step_result.get("selection_quality", "Unknown")
                    if quality in quality_counts:
                        quality_counts[quality] += 1

            # Calculate overall statistics
            overall_score = sum(all_scores) / len(all_scores) if all_scores else None
            suboptimal_count = quality_counts["Suboptimal"] + quality_counts["Poor"] + quality_counts["Hallucinated"]

            results["metric_3_1_tool_selection_quality"] = {
                "overall_score": overall_score,
                "total_actions_in_trace": total_actions,
                "actions_evaluated": len(sampled_indices),
                "sample_rate": f"every {sample_rate}th action",
                "suboptimal_selections": suboptimal_count,
                "quality_distribution": quality_counts,
                "steps": all_step_results,
                "method": "llm_as_a_judge"
            }
        else:
            results["metric_3_1_tool_selection_quality"] = {
                "overall_score": None,
                "total_actions_in_trace": 0,
                "actions_evaluated": 0,
                "steps": [],
                "note": "No action steps found",
                "method": "llm_as_a_judge"
            }
    else:
        results["metric_3_1_tool_selection_quality"] = "Skipped (No LLM Client)"

    # Metrik 3.2: Tool Execution Success (Judge - Sampled)
    if client:
        # Find all action-observation pairs, then sample every Nth
        action_obs_pairs = []
        for i, s in enumerate(trace.steps):
            if s.type == "action" and i + 1 < len(trace.steps) and trace.steps[i + 1].type == "observation":
                action_obs_pairs.append((i, i + 1))  # (action_idx, observation_idx)

        sampled_pairs = action_obs_pairs[::sample_rate]

        total_pairs = len(action_obs_pairs)
        print(f"Running 3.2 Tool Execution Success (LLM) - {len(sampled_pairs)} of {total_pairs} action-observation pairs (every {sample_rate}th)...")

        if sampled_pairs:
            all_step_results = []
            success_count = 0
            failure_count = 0
            failure_categories = {"Syntax Error": 0, "Command Not Found": 0, "Crash/Exception": 0, "Timeout": 0, "Permission Denied": 0, "Other_Misuse": 0}

            for pair_num, (action_idx, obs_idx) in enumerate(sampled_pairs):
                action_step = trace.steps[action_idx]
                obs_step = trace.steps[obs_idx]
                action_sequence_num = action_obs_pairs.index((action_idx, obs_idx)) + 1  # 1-based

                # Get action content (up to 500 chars)
                action_content = action_step.content[:500]
                if len(action_step.content) > 500:
                    action_content += "..."

                # Get observation - smart truncation: keep start and end for error visibility
                obs_content = obs_step.content
                if len(obs_content) > 1500:
                    # Keep first 750 chars and last 500 chars (errors often at end)
                    obs_content = obs_content[:750] + f"\n\n... [{len(obs_step.content)} chars total, truncated] ...\n\n" + obs_content[-500:]

                step_result = call_judge(
                    client,
                    PROMPTS["3.2"].format(
                        step_id=action_sequence_num,
                        action=action_content,
                        obs=obs_content
                    ),
                    metric_name=f"metric_3.2_action_{action_sequence_num}_trace_step_{action_idx}"
                )

                if isinstance(step_result, dict):
                    # Add clear identification
                    step_result["action_number"] = action_sequence_num
                    step_result["trace_step_id"] = action_idx
                    if "action_evaluated" not in step_result:
                        step_result["action_evaluated"] = action_step.content[:100] + ("..." if len(action_step.content) > 100 else "")

                    all_step_results.append(step_result)

                    # Count successes and failures
                    if step_result.get("success") is True:
                        success_count += 1
                    else:
                        failure_count += 1
                        category = step_result.get("failure_category", "Other_Misuse")
                        if category in failure_categories:
                            failure_categories[category] += 1
                        elif category != "None":
                            failure_categories["Other_Misuse"] += 1

            # Calculate success rate
            total_evaluated = success_count + failure_count
            success_rate = success_count / total_evaluated if total_evaluated > 0 else None

            results["metric_3_2_tool_execution_success"] = {
                "success_rate": success_rate,
                "total_pairs_in_trace": total_pairs,
                "pairs_evaluated": len(sampled_pairs),
                "sample_rate": f"every {sample_rate}th pair",
                "successes": success_count,
                "failures": failure_count,
                "failure_breakdown": failure_categories,
                "steps": all_step_results,
                "method": "llm_as_a_judge"
            }
        else:
            results["metric_3_2_tool_execution_success"] = {
                "success_rate": None,
                "total_pairs_in_trace": 0,
                "pairs_evaluated": 0,
                "steps": [],
                "note": "No action-observation pairs found",
                "method": "llm_as_a_judge"
            }
    else:
        results["metric_3_2_tool_execution_success"] = "Skipped (No LLM Client)"

    # Metrik 3.3: Tool Usage Efficiency (Deterministisch with 3.1 integration)
    print("Running 3.3 Tool Usage Efficiency (Deterministic with 3.1 weighting)...")
    metric_3_1_data = results.get("metric_3_1_tool_selection_quality") if isinstance(results.get("metric_3_1_tool_selection_quality"), dict) else None
    results["metric_3_3_tool_usage_efficiency"] = calculate_metric_3_3_tool_usage_efficiency(trace, metric_3_1_data, CONTEXT_WINDOW_SIZE)

    # Kategorie 4: Wissen (Das "Gedächtnis")
    # Metrik 4.1: Context Utilization & Consistency (Judge - Sliding Window Sampling)
    if client:
        # Get steps excluding system messages
        eval_steps = [s for s in trace.steps if s.role != "system"]

        # Identify sample points: prefer steps with thoughts, sample every Nth step
        # Start sampling after we have enough steps for a window
        sample_points = []
        for i in range(context_window_steps, len(eval_steps), context_sample_rate):
            # Prefer steps that have a thought (more visible reasoning)
            step = eval_steps[i]
            if step.type == "thought" or step.type == "action":
                sample_points.append(i)

        # If no good sample points, fall back to regular interval
        if not sample_points and len(eval_steps) > context_window_steps:
            sample_points = list(range(context_window_steps, len(eval_steps), context_sample_rate))

        num_samples = len(sample_points)
        print(f"Running 4.1 Context Utilization & Consistency (LLM) - {num_samples} sample point(s) with window size {context_window_steps}...")

        if sample_points:
            all_window_results = []
            all_scores = []
            total_flaws = 0

            for sample_idx, current_step_idx in enumerate(sample_points):
                # Extract sliding window: [current - window_size, current]
                start_idx = max(0, current_step_idx - context_window_steps + 1)
                window_steps = eval_steps[start_idx:current_step_idx + 1]

                # Build window log with head-tail truncation for observations
                window_log_parts = []
                for ws in window_steps:
                    content = ws.content.strip()
                    # Apply head-tail truncation for observations (max ~2000 chars)
                    if ws.type == "observation" and len(content) > 2000:
                        content = content[:1000] + f"\n\n... [{len(ws.content)} chars total, truncated] ...\n\n" + content[-800:]
                    elif len(content) > 1000:
                        content = content[:500] + f" ... [{len(ws.content)} chars truncated] ... " + content[-300:]

                    window_log_parts.append(f"[Step {ws.step_id}] {ws.role.upper()} ({ws.type}): {content}")

                window_log = "\n\n".join(window_log_parts)

                # Get actual step IDs for the prompt
                start_step_id = window_steps[0].step_id if window_steps else start_idx
                current_step_id = window_steps[-1].step_id if window_steps else current_step_idx

                window_result = call_judge(
                    client,
                    PROMPTS["4.1"].format(
                        task=task_desc,
                        window_size=context_window_steps,
                        start_step=start_step_id,
                        current_step=current_step_id,
                        log=window_log
                    ),
                    metric_name=f"metric_4.1_window_{sample_idx + 1}_steps_{start_step_id}_to_{current_step_id}"
                )

                if isinstance(window_result, dict):
                    window_result["window_start_step"] = start_step_id
                    window_result["window_end_step"] = current_step_id
                    all_window_results.append(window_result)

                    if window_result.get("score") is not None:
                        all_scores.append(window_result["score"])

                    flaws = window_result.get("flaws_detected", [])
                    if isinstance(flaws, list):
                        total_flaws += len(flaws)

            # Aggregate results
            overall_score = sum(all_scores) / len(all_scores) if all_scores else None

            results["metric_4_1_context_utilization_consistency"] = {
                "overall_score": overall_score,
                "total_flaws_detected": total_flaws,
                "windows_evaluated": len(sample_points),
                "window_size": context_window_steps,
                "sample_rate": f"every {context_sample_rate}th step",
                "windows": all_window_results,
                "method": "llm_as_a_judge_sliding_window"
            }
        else:
            results["metric_4_1_context_utilization_consistency"] = {
                "overall_score": None,
                "total_flaws_detected": 0,
                "windows_evaluated": 0,
                "note": f"Not enough steps for sliding window (need > {context_window_steps} steps, have {len(eval_steps)})",
                "method": "llm_as_a_judge_sliding_window"
            }
    else:
        results["metric_4_1_context_utilization_consistency"] = "Skipped (No LLM Client)"

    # Kategorie 5: Multi-Agenten (MAS)
    # Metrik 5.1: Communication Efficiency (Judge - MAS only)
    if trace.is_multi_agent:
        print("Running 5.1 Communication Efficiency (LLM - MAS only)...")
    results["metric_5_1_communication_efficiency"] = calculate_metric_5_1_communication_efficiency(trace, client, task_desc)

    # Metrik 5.2: Information Diversity (Deterministisch - Embeddings - MAS only)
    results["metric_5_2_information_diversity"] = calculate_metric_5_2_information_diversity(trace, client)

    # Metrik 5.3: Path Redundancy (Deterministisch - MAS only)
    results["metric_5_3_unique_path_redundancy"] = calculate_metric_5_3_path_redundancy(trace)

    # Metrik 5.4: Agent Invocation Distribution (Deterministisch - MAS only)
    results["metric_5_4_agent_invocation_distribution"] = calculate_metric_5_4_agent_invocation(trace)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"eval_{agent_type_arg}_{trace.task_id}.json")
    with open(outfile, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved to {outfile}")

    # Save API call logs
    if API_CALL_LOGS:
        log_file = os.path.join(OUTPUT_DIR, f"api_logs_{agent_type_arg}_{trace.task_id}.json")
        log_summary = {
            "agent": agent_type_arg,
            "task": trace.task_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_calls": len(API_CALL_LOGS),
            "total_duration_seconds": round(sum(l.get("duration_seconds", 0) for l in API_CALL_LOGS), 3),
            "total_tokens": sum(l.get("total_tokens", 0) for l in API_CALL_LOGS),
            "calls": API_CALL_LOGS
        }
        with open(log_file, "w") as f: json.dump(log_summary, f, indent=2)
        print(f"API logs saved to {log_file} ({len(API_CALL_LOGS)} calls, {log_summary['total_tokens']} tokens)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--agent", default="OpenHands", help="Name of the agent system (e.g. OpenHands, MetaGPT)")
    parser.add_argument("--mas", action="store_true", help="Force MAS evaluation")
    parser.add_argument("--sample-rate", type=int, default=5, help="Sample rate for metrics 3.1 and 3.2 (evaluate every Nth action, default: 5)")
    parser.add_argument("--context-window-steps", type=int, default=8, help="Sliding window size for metric 4.1 (number of steps to include, default: 8)")
    parser.add_argument("--context-sample-rate", type=int, default=4, help="Sample rate for metric 4.1 (evaluate every Nth step, default: 4)")
    args = parser.parse_args()
    main(args.file_path, args.agent, args.mas, args.sample_rate, args.context_window_steps, args.context_sample_rate)
