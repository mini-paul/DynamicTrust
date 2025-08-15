# graph_state.py
from typing import List, Dict, TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Represents the state of our multi-agent system.
    """
    query: str
    agents: List[str]

    # A dictionary mapping agent names to their credibility scores (CrS)
    credibility_scores: Dict[str, float]

    # A dictionary to hold the output of each agent for the current round
    round_outputs: Dict[str, str]

    # The aggregated answer from the previous round, used for CSc calculation
    aggregated_answer: str

    # The final, definitive answer after all iterations
    final_answer: str

    # Counter for the number of iterations
    iteration_count: int

    # A list to maintain the history of messages for context
    messages: Annotated[list, add_messages]