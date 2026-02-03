from pathlib import Path
from typing import Any, Dict

from src.llm import GeminiClient, LLMConfig
from src.sandbox.local_runner import run_python_code
from src.utils.config import load_yaml

from .state import AgentState


def _append_history(state: AgentState, event: Dict[str, Any]) -> None:
    history = state.setdefault("history", [])
    history.append(event)


def _load_prompts() -> Dict[str, str]:
    prompts = load_yaml("configs/prompts.yaml")
    return {
        "planner": prompts.get("planner", ""),
        "coder": prompts.get("coder", ""),
        "reviewer": prompts.get("reviewer", ""),
    }


def _load_agent_config(role: str) -> Dict[str, Any]:
    base_cfg = load_yaml("configs/config.yaml").get("llm", {})
    agent_cfg = load_yaml("configs/agents.yaml").get(role, {})
    return {
        "model": agent_cfg.get("model", base_cfg.get("model", "gemini-1.5-flash")),
        "temperature": agent_cfg.get("temperature", base_cfg.get("temperature", 0.2)),
        "max_tokens": agent_cfg.get("max_tokens"),
    }


def _build_client(role: str) -> GeminiClient:
    cfg = _load_agent_config(role)
    return GeminiClient(
        LLMConfig(
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
    )


def _format_context(state: AgentState) -> str:
    lines = [
        f"input_file: {state.get('input_file', '')}",
        f"target_column: {state.get('target_column', '')}",
        f"problem_type: {state.get('problem_type', '')}",
        f"iter_count: {state.get('iter_count', 0)}",
    ]
    if state.get("plan"):
        lines.append(f"plan: {state['plan']}")
    if state.get("execution_result"):
        lines.append(f"execution_result: {state['execution_result']}")
    return "\n".join(lines)


def plan_step(state: AgentState) -> AgentState:
    prompts = _load_prompts()
    client = _build_client("planner")
    context = _format_context(state)
    user_prompt = f"""Context:\n{context}\n\nReturn a concise analysis plan."""
    response = client.generate_text(prompts["planner"], user_prompt)
    if response:
        state["plan"] = response
    _append_history(state, {"step": "plan", "plan": state.get("plan", "")})
    return state


def code_gen_step(state: AgentState) -> AgentState:
    prompts = _load_prompts()
    client = _build_client("coder")
    context = _format_context(state)
    user_prompt = f"""Context:\n{context}\n\nGenerate code for preprocessing and proxy training."""
    response = client.generate_text(prompts["coder"], user_prompt)
    if response:
        state["generated_code"] = response
    _append_history(state, {"step": "code_gen"})
    return state


def execute_step(state: AgentState) -> AgentState:
    config = state.get("config", {})
    agent_cfg = config.get("agent", {})
    paths_cfg = config.get("paths", {})

    timeout_sec = int(agent_cfg.get("timeout_sec", 300))
    execute_enabled = bool(agent_cfg.get("execute_enabled", True))
    scripts_dir = paths_cfg.get("generated_scripts", "generated/scripts")

    Path(scripts_dir).mkdir(parents=True, exist_ok=True)

    if state.get("generated_code"):
        pipeline_path = Path(scripts_dir) / "pipeline.py"
        pipeline_path.write_text(state["generated_code"], encoding="utf-8")

    if execute_enabled and state.get("generated_code"):
        result = run_python_code(state["generated_code"], workdir=str(scripts_dir), timeout_sec=timeout_sec)
        state["last_run"] = result
        state["execution_result"] = (
            f"exit_code={result.get('exit_code')} duration={result.get('duration_sec')}"
        )
    else:
        state["execution_result"] = "Execution skipped (disabled or no code)."

    state["iter_count"] = int(state.get("iter_count", 0)) + 1
    _append_history(state, {"step": "execute", "result": state["execution_result"]})
    return state


def review_step(state: AgentState) -> AgentState:
    prompts = _load_prompts()
    client = _build_client("reviewer")
    context = _format_context(state)
    user_prompt = f"""Context:\n{context}\n\nDecide whether to iterate or stop."""
    response = client.generate_text(prompts["reviewer"], user_prompt)
    if response:
        state["plan"] = response
    max_iters = int(state.get("max_iters", 1))
    iter_count = int(state.get("iter_count", 0))
    state["stop"] = iter_count >= max_iters
    _append_history(state, {"step": "review", "stop": state["stop"]})
    return state
