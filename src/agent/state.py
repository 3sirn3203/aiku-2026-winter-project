from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    config: Dict[str, Any]
    input_file: str
    target_column: str
    problem_type: str
    plan: str
    generated_code: str
    execution_result: str
    last_run: Dict[str, Any]
    history: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    best_config: Dict[str, Any]
    iter_count: int
    max_iters: int
    stop: bool
