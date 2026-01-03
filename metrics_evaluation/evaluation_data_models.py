from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class StandardStep:
    step_id: int
    role: str
    type: str
    content: str
    metadata: Dict[str, Any]
    agent_name: str = "default"

@dataclass
class EvaluationTrace:
    agent_name: str
    task_id: str
    steps: List[StandardStep]
    total_cost: float
    total_tokens: int
    duration_seconds: float
    success_status: Optional[bool]
    is_multi_agent: bool = False
