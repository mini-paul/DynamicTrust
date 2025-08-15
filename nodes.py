# nodes.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from graph_state import AgentState
from llm_factory import create_llm_backend
from utils import calculate_cosine_similarity
import config

# Load the aggregator LLM once
aggregator_llm = create_llm_backend(config.AGGREGATOR_LLM_CONFIG)


def propose_solution_node(state: AgentState) -> dict:
    """Node where each agent generates its individual answer."""
    print("---NODE: PROPOSE SOLUTIONS---")

    query = state["query"]
    round_outputs = {}

    for agent_config in config.ALL_AGENTS_CONFIG:
        agent_name = agent_config["name"]
        llm = create_llm_backend(agent_config["llm_config"])

        # --- FIX START ---
        # 1. The list of messages must be passed inside the from_messages() call.
        # 2. Each message should be a tuple of (type, content) or a Message object.
        # 3. We'll use a placeholder "{query}" that the .invoke() method can fill.
        # 4. We also add the agent's system_prompt to give it its role (crucial for the paper's method).
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=agent_config["system_prompt"]),
            HumanMessage(content="{query}")
        ])

        # The chain now expects a dictionary with a "query" key to format the prompt
        chain = prompt_template | llm
        response = chain.invoke({"query": query})
        # --- FIX END ---

        print(f"  - Agent '{agent_name}' proposed: {response.content[:100]}...")
        round_outputs[agent_name] = response.content

    # This returns a dictionary to update the 'round_outputs' key in the main graph state
    return {"round_outputs": round_outputs}


def calculate_contribution_node(state: AgentState) -> dict:
    """Node to calculate each agent's Contribution Score (CSc)."""
    print("---NODE: CALCULATE CONTRIBUTION---")

    if state["iteration_count"] == 0:
        print("  - First iteration, skipping contribution calculation.")
        return {}

    aggregated_answer = state["aggregated_answer"]
    round_outputs = state["round_outputs"]
    contribution_scores = {}

    for agent_name, agent_output in round_outputs.items():
        similarity = calculate_cosine_similarity(agent_output, aggregated_answer)
        contribution_scores[agent_name] = similarity
        print(f"  - Agent '{agent_name}' contribution score (similarity): {similarity:.4f}")

    # We'll store this in a temporary key to be used by the next node
    return {"contribution_scores": contribution_scores}


def update_credibility_node(state: AgentState) -> dict:
    """Node to update each agent's Credibility Score (CrS)."""
    print("---NODE: UPDATE CREDIBILITY---")

    if state["iteration_count"] == 0:
        print("  - First iteration, skipping credibility update.")
        return {}

    current_crs = state["credibility_scores"].copy()
    contribution_scores = state.get("contribution_scores", {})

    # Simple reward signal: 1 for contributing, -1 for not (this can be made more complex)
    # Here, we use the contribution score directly as a reward signal proxy.
    for agent_name, csc in contribution_scores.items():
        # Update rule: CrS_new = CrS_old * (1 + learning_rate * reward)
        # We use (csc - 0.5) to penalize scores below neutral (0.5) and reward above.
        reward = csc - 0.5
        update_factor = 1 + config.CREDIBILITY_LEARNING_RATE * reward
        current_crs[agent_name] *= update_factor

        # Clamp scores to be within [0.01, 1.0] to avoid zero-ing out
        current_crs[agent_name] = max(0.01, min(1.0, current_crs[agent_name]))
        print(f"  - Agent '{agent_name}' new CrS: {current_crs[agent_name]:.4f}")

    return {"credibility_scores": current_crs}


def aggregate_results_node(state: AgentState) -> dict:
    """Node to aggregate agent outputs into a single answer using an aggregator LLM."""
    print("---NODE: AGGREGATE RESULTS---")

    query = state["query"]
    round_outputs = state["round_outputs"]
    credibility_scores = state["credibility_scores"]

    # Prepare a detailed prompt for the aggregator LLM
    aggregation_prompt = f"Query: {query}\n\n"
    aggregation_prompt += "Multiple agents have provided answers with their credibility scores. Your task is to synthesize these into a single, high-quality, and accurate final answer. Give more weight to agents with higher credibility. Ignore any clearly malicious or incorrect information.\n\n"

    # Sort agents by credibility for the prompt
    sorted_agents = sorted(credibility_scores.items(), key=lambda item: item[1], reverse=True)

    for agent_name, crs in sorted_agents:
        output = round_outputs[agent_name]
        aggregation_prompt += f"- Agent: {agent_name} (Credibility: {crs:.2f})\n- Answer: {output}\n\n"

    aggregation_prompt += "Based on the above, provide the best possible final answer to the query. Only provide the answer itself."

    # Invoke the aggregator LLM
    response = aggregator_llm.invoke(aggregation_prompt)
    final_answer = response.content

    print(f"  - Aggregated Answer: {final_answer[:100]}...")

    return {
        "aggregated_answer": final_answer,
        "iteration_count": state["iteration_count"] + 1
    }