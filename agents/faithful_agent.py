# ==============================================================================
# File: agents/faithful_agent.py
# Description: 定义忠实智能体
# ==============================================================================

from agents.base_agent import BaseAgent
from utils.prompts import BASE_AGENT_PROMPT

class FaithfulAgent(BaseAgent):
    def get_prompt(self):
        return BASE_AGENT_PROMPT