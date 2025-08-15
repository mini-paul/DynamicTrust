from graph_builder import build_graph
import config
# Assuming you have a message class, like AIMessage, HumanMessage, etc.
# If not, an empty list is sufficient.
from langchain_core.messages import BaseMessage


def main():
    """
    Main function to run the adversary-resistant multi-agent system.
    """
    # Build the compiled LangGraph application
    app = build_graph()

    # Define the initial query
    query = "What was the primary cause of the fall of the Roman Empire? Provide a detailed explanation."

    # --- FIX START ---
    # Initialize the state for the graph
    initial_state = {
        "query": query,
        # 1. Removed extra closing bracket ']'
        "agents": [agent["name"] for agent in config.ALL_AGENTS_CONFIG],
        # 2. Removed extra curly braces '{}' around the dictionary comprehension
        "credibility_scores": {agent["name"]: 0.5 for agent in config.ALL_AGENTS_CONFIG},
        "round_outputs": {},
        "aggregated_answer": "",
        "final_answer": "",
        "iteration_count": 0,
        # 3. Initialized 'messages' to an empty list
        "messages": [],
    }
    # --- FIX END ---

    print("\n--- STARTING AGENT WORKFLOW ---")
    print(f"Query: {query}\n")

    # Invoke the graph with the initial state
    # The 'recursion_limit' is important for graphs with loops to prevent infinite execution
    final_state = app.invoke(initial_state, config={"recursion_limit": 10})

    print("\n--- AGENT WORKFLOW COMPLETE ---")
    print(f"\nFinal Credibility Scores:")
    for agent, score in final_state['credibility_scores'].items():
        print(f"  - {agent}: {score:.4f}")

    print(f"\nFinal Answer:\n{final_state['final_answer']}")


if __name__ == "__main__":
    main()