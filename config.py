# config.py

# Configuration for the LLM that will act as the final aggregator/judge
AGGREGATOR_LLM_CONFIG = {
    "provider": "deepseek",
    "model_name": "deepseek-chat",
    "temperature": 0.1
}

# --- Agent Definitions ---
# Each agent has a name, an LLM configuration, and a system prompt defining its role.

# A standard, helpful agent
AGENT_1_CONFIG = {
    "name": "Helpful_Agent_1",
    "llm_config": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "temperature": 0.7
    },
    "system_prompt": "You are a helpful assistant. Provide a clear, accurate, and concise answer to the user's query."
}

# Another standard, helpful agent
AGENT_2_CONFIG = {
    "name": "Helpful_Agent_2",
    "llm_config": {
        "provider": "ollama", # Example using a local model
        "model_name": "llama3",
        "temperature": 0.7
    },
    "system_prompt": "You are a knowledgeable expert. Your goal is to give a detailed and well-structured response."
}

# An adversarial agent programmed to be subtly incorrect
ADVERSARIAL_AGENT_1_CONFIG = {
    "name": "Adversary_Subtle",
    "llm_config": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "temperature": 0.9
    },
    "system_prompt": "You are an adversarial agent. Your goal is to provide an answer that seems plausible but contains a subtle, critical error. Try to be persuasive."
}

# An adversarial agent programmed to be confidently wrong
ADVERSARIAL_AGENT_2_CONFIG = {
    "name": "Adversary_Confident",
    "llm_config": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "temperature": 0.8
    },
    "system_prompt": "You are a contrarian agent. Your task is to confidently provide an answer that is incorrect, but argue for it as if it were fact. Be very convincing."
}

# List of all agents to be used in the graph
ALL_AGENTS_CONFIG = [AGENT_1_CONFIG,ADVERSARIAL_AGENT_1_CONFIG,ADVERSARIAL_AGENT_2_CONFIG]

# --- Graph Settings ---
MAX_ITERATIONS = 3
CREDIBILITY_LEARNING_RATE = 0.1