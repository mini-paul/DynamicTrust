# graph_builder.py
from langgraph.graph import StateGraph, END
from graph_state import AgentState
import nodes
import config

def should_continue(state: AgentState) -> str:
    """
    Routing function to decide whether to continue iterating or end.
    """
    print("---ROUTER: SHOULD CONTINUE?---")
    iteration_count = state["iteration_count"]
    if iteration_count >= config.MAX_ITERATIONS:
        print(f"  - Reached max iterations ({config.MAX_ITERATIONS}). Ending.")
        return "end"
    else:
        print(f"  - Iteration {iteration_count}. Continuing.")
        return "continue"

def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph for the multi-agent system.
    """
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("propose", nodes.propose_solution_node)
    workflow.add_node("calculate_contribution", nodes.calculate_contribution_node)
    workflow.add_node("update_credibility", nodes.update_credibility_node)
    workflow.add_node("aggregate", nodes.aggregate_results_node)

    # Define the graph's flow
    workflow.set_entry_point("propose")
    workflow.add_edge("propose", "calculate_contribution")
    workflow.add_edge("calculate_contribution", "update_credibility")
    workflow.add_edge("update_credibility", "aggregate")

    # Add the conditional edge for the iterative loop
    workflow.add_conditional_edges(
        "aggregate",
        should_continue,
        {
            "continue": "propose",  # If continue, loop back to the proposal node
            "end": END              # If end, exit the graph
        }
    )

    # Compile the graph into a runnable application
    app = workflow.compile()
    print("Graph compiled successfully.")
    return app