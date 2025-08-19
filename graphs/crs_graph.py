# ==============================================================================
# File: graphs/crs_graph.py
# Description: 定义核心的CrS协作图状态和图节点
# ==============================================================================

import operator
from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage
import numpy as np
from config import INITIAL_CRS, LEARNING_RATE


# --- Graph State Definition ---
class CrsGraphState(TypedDict):
    query: str
    ground_truth: str
    agent_team: Dict  # Holds agent instances
    credibility_scores: Dict[str, float]

    # Intermediate states for the graph flow
    iteration: int
    conversation_history: List[BaseMessage]
    agent_outputs: Dict[str, str]
    final_answer: str

    # Final evaluation states
    reward: float
    contribution_scores: Dict[str, float]


# --- Graph Node Functions ---

def initialize_crs(state: CrsGraphState) -> CrsGraphState:
    """
    Node to initialize credibility scores for all agents in the team.
    """
    agent_ids = list(state["agent_team"].keys())
    state["credibility_scores"] = {agent_id: INITIAL_CRS for agent_id in agent_ids}
    state["iteration"] = 0
    state["conversation_history"] = []
    print("--- CREDIBILITY SCORES INITIALIZED ---")
    print(state["credibility_scores"])
    return state


def agent_collaboration_step(state: CrsGraphState) -> CrsGraphState:
    """
    Node for a single round of agent collaboration. Each agent gives an answer.
    """
    query = state["query"]
    agent_team = state["agent_team"]

    # Format conversation history for the prompt
    history_str = "\n".join([f"{msg.name}: {msg.content}" for msg in state["conversation_history"]])

    current_outputs = {}
    new_history_messages = []

    print(f"\n--- COLLABORATION ITERATION {state['iteration'] + 1} ---")

    for agent_id, agent in agent_team.items():
        response = agent.invoke(query, history_str)
        current_outputs[agent_id] = response
        new_history_messages.append(HumanMessage(content=response, name=agent_id))
        print(f"  - {agent_id} answered.")

    state["agent_outputs"] = current_outputs
    state["conversation_history"].extend(new_history_messages)
    state["iteration"] += 1

    return state


def crs_aware_aggregation(state: CrsGraphState) -> CrsGraphState:
    """
    Node to aggregate agent outputs using a simplified CrS-weighted approach.
    This is a simplified version of the "centroid-based" aggregation.
    """
    agent_outputs = state["agent_outputs"]
    credibility_scores = state["credibility_scores"]

    if not agent_outputs:
        state["final_answer"] = "No consensus could be reached."
        return state

    # Simple weighted random choice based on CrS
    agents = list(agent_outputs.keys())
    scores = np.array([credibility_scores[agent] for agent in agents])

    # Handle case where all scores are zero
    if scores.sum() == 0:
        probabilities = np.ones(len(agents)) / len(agents)
    else:
        probabilities = scores / scores.sum()

    chosen_agent = np.random.choice(agents, p=probabilities)
    final_answer = agent_outputs[chosen_agent]

    state["final_answer"] = final_answer
    print("\n--- AGGREGATION COMPLETE ---")
    print(f"Scores: { {k: round(v, 2) for k, v in credibility_scores.items()} }")
    print(f"Probabilities: { {agents[i]: round(probabilities[i], 2) for i in range(len(agents))} }")
    print(f"Chosen Agent: {chosen_agent}")
    print(f"Final Answer: {final_answer}")

    return state


def evaluate_and_reward(state: CrsGraphState) -> CrsGraphState:
    """
    Node where the Judge Agent evaluates the final answer and assigns rewards.
    """
    judge = state["agent_team"]["judge"]  # Assuming judge is passed in the team
    query = state["query"]
    final_answer = state["final_answer"]
    ground_truth = state["ground_truth"]

    # 1. Get reward score
    reward = judge.get_reward_score(query, final_answer, ground_truth)
    state["reward"] = reward
    print("\n--- EVALUATION & REWARD ---")
    print(f"Reward Score: {reward}")

    # 2. Get contribution scores
    history_str = "\n".join([f"{msg.name}: {msg.content}" for msg in state["conversation_history"]])
    agent_outputs_str = "\n".join([f"{agent_id}: {output}" for agent_id, output in state["agent_outputs"].items()])
    full_context = f"--- Conversation History ---\n{history_str}\n\n--- Final Agent Outputs ---\n{agent_outputs_str}"

    # Exclude judge from contribution scoring
    agent_ids_for_scoring = [id for id in state["agent_team"].keys() if id != 'judge']

    # Re-format context to only include scorable agents for the judge
    scorable_agent_context = "\n".join(
        [f"{agent_id}: {state['agent_outputs'][agent_id]}" for agent_id in agent_ids_for_scoring])

    contribution_scores = judge.get_contribution_scores(query, final_answer, scorable_agent_context)
    state["contribution_scores"] = contribution_scores
    print(f"Contribution Scores: {contribution_scores}")

    return state


def update_crs(state: CrsGraphState) -> CrsGraphState:
    """
    Node to update the credibility scores based on reward and contributions.
    """
    reward = state["reward"]
    contribution_scores = state["contribution_scores"]
    current_crs = state["credibility_scores"].copy()

    print("\n--- CRS UPDATE ---")
    for agent_id, csc in contribution_scores.items():
        if agent_id in current_crs:
            old_crs = current_crs[agent_id]
            # The update formula from the paper
            new_crs = old_crs * (1 + LEARNING_RATE * csc * reward)
            # Clip scores to be within a reasonable range [0.01, 1.0]
            new_crs = max(0.01, min(1.0, new_crs))
            current_crs[agent_id] = new_crs
            print(f"  - {agent_id}: {round(old_crs, 3)} -> {round(new_crs, 3)}")

    state["credibility_scores"] = current_crs
    return state


# --- Conditional Edges ---
def should_continue_collaboration(state: CrsGraphState) -> str:
    """
    Conditional edge to decide whether to continue collaboration or move to aggregation.
    """
    if state["iteration"] >= 1:  # For simplicity, we run only one round of collaboration
        return "aggregate"
    return "continue"