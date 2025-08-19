# ==============================================================================
# File: graphs/crs_graph.py
# Description: 定义核心的CrS协作图状态和图节点
# ==============================================================================

import operator
from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage
import numpy as np
from config import INITIAL_CRS, LEARNING_RATE

import re

# --- Graph State Definition ---
class CrsGraphState(TypedDict):
    query: str
    ground_truth: str
    agent_team: Dict  # Holds agent instances
    judge_agent: Dict
    credibility_scores: Dict[str, float]

    # Intermediate states for the graph flow
    iteration: int
    conversation_history: List[BaseMessage]
    agent_outputs: Dict[str, str]
    final_answer: str

    # Final evaluation states
    reward: float
    contribution_scores: Dict[str, float]


def extract_final_answer(response: str) -> str:
    """提取 <think> 之后的内容作为最终答案"""
    # 方法 1: 删除所有 <think> 标签内容
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # 方法 2: 提取最后一个 </think> 后的内容
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    clean_response = clean_response.replace("\n","")

    return clean_response.strip()

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

    print("initialize_crs 0000000000000000000000000 --- ", state)
    return state


def agent_collaboration_step(state: CrsGraphState) -> CrsGraphState:
    """
    Node for a single round of agent collaboration. Each agent gives an answer.
    """
    query = state["query"]
    agent_team = state["agent_team"]

    # Format conversation history for the prompt
    history_str = "\n".join([f"{msg.name}: {msg.content}" for msg in state.get("conversation_history",[])])

    current_outputs = {}
    new_history_messages = []

    print(f"\n--- COLLABORATION ITERATION {state.get("iteration",0) + 1} ---")

    for agent_id, agent in agent_team.items():
        print(f"{agent_id}: {agent}")
        response = agent.invoke(query, history_str)
        only_response = extract_final_answer(response)

        current_outputs[agent_id] = only_response
        new_history_messages.append(HumanMessage(content=only_response, name=agent_id))
        print(f"  - {agent_id} answered.")
        print(current_outputs)
        print(history_str)
        print("*********************************************")
    state["agent_outputs"] = current_outputs
    state["conversation_history"] = state.get("conversation_history",[]).extend(new_history_messages)
    state["iteration"] = state.get("iteration",0) + 1
    print("agent_collaboration_step 55555555555555555555555555555 --- ", state)

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

    print("crs_aware_aggregation 111111111111111 --- ",state)

    return state


def evaluate_and_reward(state: CrsGraphState) -> CrsGraphState:
    """
    Node where the Judge Agent evaluates the final answer and assigns rewards.
    """

    print("evaluate_and_reward 2222222222222222222222 --- ", state)
    # judge = state["agent_team"]["judge"]  # Assuming judge is passed in the team
    judge = state["judge_agent"]

    query = state["query"]

    final_answer = state["final_answer"]
    ground_truth = state["ground_truth"]

    # 1. Get reward score
    reward = judge.get_reward_score(query, final_answer, ground_truth)
    state["reward"] = reward
    print("\n--- EVALUATION & REWARD ---")
    print(f"Reward Score: {reward}")

    # 2. Get contribution scores
    history = state.get("conversation_history",[])
    print("##################################111111111111111111111")
    print(history)
    history_str = ""
    if history is  None:
        history_str = ""
    else:
        history_str = "\n".join([f"{msg.name}: {msg.content}" for msg in state.get("conversation_history",[])])
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

    print("update_crs 3333333333333333333333333 --- ", state)
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
    print("should_continue_collaboration 44444444444444444444444 --- ", state)
    if state["iteration"] >= 1:  # For simplicity, we run only one round of collaboration
        return "aggregate"
    return "continue"