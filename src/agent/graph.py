from langgraph.graph import END, StateGraph

from .nodes import plan_step, code_gen_step, execute_step, review_step
from .state import AgentState


def _should_continue(state: AgentState) -> str:
    if state.get("stop"):
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("plan_step", plan_step)
    graph.add_node("code_gen_step", code_gen_step)
    graph.add_node("execute_step", execute_step)
    graph.add_node("review_step", review_step)

    graph.set_entry_point("plan_step")
    graph.add_edge("plan_step", "code_gen_step")
    graph.add_edge("code_gen_step", "execute_step")
    graph.add_edge("execute_step", "review_step")
    graph.add_conditional_edges(
        "review_step",
        _should_continue,
        {
            "continue": "plan_step",
            "end": END,
        },
    )
    return graph.compile()
